#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/Hashing.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/utils.hpp"
#include "potato/util/common.hpp"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <string>

namespace potato::analysis {
    struct stg_elem {
        std::optional< pt_element > elem;
        std::optional< size_t > dummy_id;

        stg_elem(mlir_value val)             : elem(pt_element(val)), dummy_id(std::nullopt) {}
        stg_elem(size_t id)                  : elem(std::nullopt), dummy_id(id) {}
        stg_elem()                           : elem(std::nullopt), dummy_id(std::nullopt) {}

        stg_elem(pt_element::elem_kind kind, mlir_value val, mlir_operation *op)
            : elem(pt_element(kind, val, op)), dummy_id(std::nullopt) {}

        static stg_elem make_func(mlir_operation *op) {
            return stg_elem(pt_element::elem_kind::func, {}, op);
        }

        static stg_elem make_glob(mlir_operation *op) {
            return stg_elem(pt_element::elem_kind::global, {}, op);
        }

        static stg_elem make_alloca(mlir_operation *op, mlir_value val = {}) {
            return stg_elem(pt_element::elem_kind::alloca, val, op);
        }

        static stg_elem make_var(mlir_value val) {
            return stg_elem(pt_element::elem_kind::var, val, nullptr);
        }

        bool operator==(const stg_elem &rhs) const = default;

        inline bool is_unknown() const { return !is_dummy() && !elem.has_value(); }
        inline bool is_dummy() const { return dummy_id.has_value(); }
        inline bool is_top() const { return is_unknown(); }
        inline bool is_bottom() const { return is_dummy(); }
        inline bool is_alloca() const { return elem && elem->is_alloca(); }
        inline bool is_var() const { return elem && elem->is_var(); }
        inline bool is_func() const { return elem && elem->is_func(); }
        inline bool is_global() const { return elem && elem->is_global(); }

        void print(llvm::raw_ostream &os) const {
            if (elem) {
                elem->print(os);
                return;
            }
            if (dummy_id) {
                os << "dummy_" << *dummy_id;
                return;
            }
            if (is_top()) {
                os << "top";
                return;
            }
        }
    };

} // namespace potato::analysis

template <>
struct std::hash< potato::analysis::stg_elem > {
    using stg_elem = potato::analysis::stg_elem;
    using pt_element = potato::analysis::pt_element;

    std::size_t operator() (const stg_elem &value) const {
        return llvm::hash_combine(std::hash< std::optional< pt_element > >{}(value.elem),
                                  std::hash< std::optional< size_t > >{}(value.dummy_id));
    }
};

namespace potato::analysis {
    namespace detail {
        template< typename T >
        struct union_find {
            std::unordered_map< T, T > parents;
            std::unordered_map< T, size_t > rank;

            std::unordered_map< T, std::unordered_set< T > > children;

            bool insert(const T& elem) {
                if (parents.find(elem) == parents.end()) {
                    parents[elem] = elem;
                    rank[elem] = 0;
                    children[elem] = {elem};
                    return true;
                }
                return false;
            };

            T find(const T &x) {
                auto find_it = parents.find(x);
                if (find_it == parents.end()) {
                    insert(x);
                    find_it = parents.find(x);
                }

                T found = x;
                T parent = find_it->second;
                while (found != parents[found]) {
                    auto &parent_parent = parents[parent];
                    parents[found] = parent_parent;
                    found = parent_parent;
                    parent = parents[found];
                }
                return found;
            }

            T set_union(const T &x, const T &y) {
                T x_root = find(x);
                T y_root = find(y);
                if (x_root == y_root)
                    return x_root;

                size_t &x_rank = rank[x_root];
                size_t &y_rank = rank[y_root];

                if (x_root.is_dummy()) {
                    parents[x_root] = y_root;
                    children[y_root].insert(x_root);
                    if (x_rank >= y_rank) {
                        y_rank = x_rank + 1;
                    }
                    return y_root;
                }
                if (y_root.is_dummy()) {
                    parents[y_root] = x_root;
                    children[x_root].insert(y_root);
                    if (y_rank >= x_rank) {
                        x_rank = y_rank + 1;
                    }
                    return x_root;
                }

                if (x_rank > y_rank) {
                    parents[y_root] = x_root;
                    children[x_root].insert(y_root);
                    return x_root;
                } else {
                    parents[x_root] = y_root;
                    children[y_root].insert(x_root);
                    if (x_rank == y_rank) {
                        y_rank++;
                    }
                    return y_root;
                }
            }
            // TODO: children iterator wrapper
        };
    } // namespace detail

    template< typename elem_t >
    struct steensgaard_info {
        detail::union_find< elem_t > sets;

        // edges between nodes(sets)
        // keep the invariant, that keys are the set roots to minimize the number of edges
        // targets can be any set members

        std::unordered_map< elem_t, elem_t > mapping;
        bool all_unknown;
        size_t dummy_count = 0;
    };

    struct steensgaard : mlir_dense_abstract_lattice {
        using mlir_dense_abstract_lattice::AbstractDenseLattice;
        using elem_t = stg_elem;
        using relation_t = steensgaard_info< elem_t >;
        std::shared_ptr< relation_t > info;

        private:
        inline auto &mapping() const { return info->mapping; }
        inline auto &sets() const { return info->sets; }
        inline auto &all_unknown() const { return info->all_unknown; }

        public:
        bool initialized() const { return (bool) info; }
        void initialize_with(std::shared_ptr< relation_t > &relation) { info = relation; }

        elem_t *lookup(const elem_t &val) {
            auto trg_it = mapping().find(val);
            if (trg_it == mapping().end())
                return &mapping().emplace(val, new_dummy()).first->second;
            return &trg_it->second;
        }

        static elem_t new_top_set() {
            // default constructor creates unknown
            return elem_t();
        }

        elem_t new_dummy() {
            return elem_t(info->dummy_count++);
        }


        static elem_t new_func(mlir_operation *op) {
            return elem_t::make_func(op);
        }

        static elem_t new_glob(mlir_operation *op) {
            return elem_t::make_glob(op);
        }

        change_result new_alloca(mlir_value val) {
            auto alloca = elem_t::make_alloca(val.getDefiningOp());
            return join_var(val, alloca);
        }

        change_result join_empty(mlir_value val) {
            return sets().insert(val) ? change_result::Change : change_result::NoChange;
        }

        change_result join_all_pointees_with(elem_t *to, const elem_t *from) {
            auto to_trg_it = mapping().find(*to);
            if (to_trg_it == mapping().end()) {
                mapping().insert({*to, *from});
                return change_result::Change;
            }
            return make_union(to_trg_it->second, *from);
        }


        change_result copy_all_pts_into(elem_t &&to, const elem_t *from) {
            auto from_rep = sets().find(*from);
            auto to_rep = sets().find(to);
            auto to_trg_it = mapping().find(to_rep);
            auto from_trg_it = mapping().find(from_rep);

            if (from_trg_it == mapping().end()) {
                if (to_trg_it == mapping().end()) {
                    // tie targets with a dummy
                    auto dummy = new_dummy();
                    sets().insert(dummy);
                    mapping().emplace(to_rep, dummy);
                    mapping().emplace(from_rep, dummy);
                } else {
                    // tie targets together
                    mapping().emplace(from_rep, to_trg_it->second);
                }
                return change_result::Change;
            }

            auto &from_trg = from_trg_it->second;

            if (to_trg_it == mapping().end()) {
                mapping().emplace(to_rep, from_trg);
                return change_result::Change;
            }

            return make_union(to_trg_it->second, from_trg);
        }

        auto add_constant(mlir_value val) { return join_empty(val); }

        change_result make_union(elem_t lhs, elem_t rhs) {
            auto lhs_root = sets().find(lhs);
            auto rhs_root = sets().find(rhs);
            if (lhs_root == rhs_root) {
                return change_result::NoChange;
            }

            auto lhs_trg = mapping().find(lhs_root);
            auto rhs_trg = mapping().find(rhs_root);

            auto new_root = sets().set_union(lhs, rhs);

            if (lhs_trg == mapping().end() && rhs_trg == mapping().end()) {
                return change_result::Change;
            }

            // merge outgoing edges
            if (lhs_trg == mapping().end()) {
                if (lhs_root == new_root) {
                    mapping().emplace(new_root, rhs_trg->second);
                    mapping().erase(rhs_trg);
                }
                return change_result::Change;
            }
            if (rhs_trg == mapping().end()) {
                if (rhs_root == new_root) {
                    mapping().emplace(new_root, lhs_trg->second);
                    mapping().erase(lhs_trg);
                }
                return change_result::Change;
            }

            if (lhs_trg->second.is_unknown()) {
                mapping()[rhs_root] = lhs_trg->second;
                return change_result::Change;
            }

            if (rhs_trg->second.is_unknown()) {
                mapping()[lhs_root] = rhs_trg->second;
            }

            // if both have outgoing edge, unify the targets and remove the redundant edge
            std::ignore = make_union(lhs_trg->second, rhs_trg->second);
            if (new_root == lhs_root) {
                mapping().erase(rhs_trg);
            } else {
                mapping().erase(lhs_trg);
            }

            return change_result::Change;
        }

        change_result join_var(const elem_t &ptr, const elem_t &new_trg) {
            auto ptr_rep = sets().find(ptr);
            auto new_trg_rep = sets().find(new_trg);

            auto ptr_trg = mapping().find(ptr_rep);
            if (ptr_trg == mapping().end()) {
                mapping().insert({ptr_rep, new_trg_rep});
                return change_result::Change;
            }
            if (ptr_trg->second.is_unknown()) {
                return change_result::NoChange;
            }
            if (new_trg_rep.is_unknown()) {
                mapping()[ptr_rep] = new_trg_rep;
                return change_result::Change;
            }
            return make_union(ptr_trg->second, new_trg_rep);
        }

        change_result join_var(const elem_t &ptr, const elem_t *new_trg) {
            return join_var(ptr, *new_trg);
        }

        change_result set_all_unknown() {
            auto &unknown = all_unknown();
            if (unknown) {
                return change_result::NoChange;
            }

            unknown = true;
            return change_result::Change;
        }

        change_result merge(const steensgaard &rhs) {
            if (info && !rhs.info)
                return change_result::NoChange;
            if (info && rhs.info == info) {
                return change_result::NoChange;
            }
            if (info && rhs.info) {
                llvm::errs() << "Merging two different relations.\n";
                //TODO
            }
            if (rhs.info) {
                info = rhs.info;
                return change_result::Change;
            }
            return change_result::NoChange;
        };

        change_result join(const mlir_dense_abstract_lattice &rhs) override {
            return this->merge(*static_cast< const steensgaard *>(&rhs));
        };

        void print(llvm::raw_ostream &os) const override {
            for (const auto &[src, trg] : mapping()) {
                src.print(os);
                os << " -> {";
                auto trg_rep = sets().find(trg);
                auto sep = "";
                for (auto child : sets().children.at(trg_rep)) {
                    os << sep;
                    child.print(os);
                    sep = ", ";
                }
                os << "}\n";
            }
        }

        static void add_dependencies(mlir::Operation *op, mlir_dense_dfa< steensgaard > *analysis, ppoint point, auto get_or_create) { return; }

        constexpr static bool propagate_assign() { return false; }
        constexpr static bool propagate_call_arg_zip() { return false; }

        alias_res alias(auto lhs, auto rhs) {
            auto lhs_trg = sets().find(*lookup(lhs));
            auto rhs_trg = sets().find(*lookup(rhs));
            if (lhs_trg.is_dummy() || rhs_trg.is_dummy()) {
                return alias_kind::NoAlias;
            }
            if (lhs_trg == rhs_trg) {
                return alias_kind::MayAlias;
            }
            if (lhs_trg != rhs_trg) {
                return alias_kind::NoAlias;
            }
            // TODO: MustAlias
        };
    };
} //namespace potato::analysis
