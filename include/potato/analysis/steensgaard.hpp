#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/Hashing.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/function_models.hpp"
#include "potato/analysis/utils.hpp"
#include "potato/util/common.hpp"

#include <memory>
#include <unordered_map>
#include <unordered_set>

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

        void print(llvm::raw_ostream &os) const;
    };

    struct fn_info {
        value_range operands;

        stg_elem res;
        bool operator==(const fn_info &rhs) const = default;
    };


    llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const stg_elem &e);
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
        std::unordered_map< elem_t, fn_info > fn_infos;
        function_models *models;
        bool all_unknown;
        size_t dummy_count = 0;
    };

    struct steensgaard : mlir_dense_abstract_lattice {
        using mlir_dense_abstract_lattice::AbstractDenseLattice;
        using elem_t = stg_elem;
        using info_t = steensgaard_info< elem_t >;
        info_t *info = nullptr;

        private:
        inline auto &mapping() const { return info->mapping; }
        inline auto &sets() const { return info->sets; }
        inline auto &all_unknown() const { return info->all_unknown; }
        change_result visit_function_model(const function_model &model, fn_interface fn, elem_t res_dummy);

        public:
        bool initialized() const { return (bool) info; }
        void initialize_with(info_t *relation) { info = relation; }

        elem_t *lookup(const elem_t &val);
        // default constructor creates unknown
        static elem_t new_top_set() { return elem_t(); }
        elem_t new_dummy();
        static elem_t new_func(mlir_operation *op) { return elem_t::make_func(op); }
        static elem_t new_glob(mlir_operation *op) { return elem_t::make_glob(op); }
        change_result new_alloca(mlir_value val);
        change_result join_empty(mlir_value val) {
            return sets().insert(val) ? change_result::Change : change_result::NoChange;
        }
        change_result join_all_pointees_with(elem_t *to, const elem_t *from);
        change_result copy_all_pts_into(elem_t &&to, const elem_t *from);

        change_result resolve_fptr_call(
            mlir_value val,
            mlir::CallOpInterface call,
            auto /*get_or_create*/,
            auto /*add_dep*/,
            auto /*propagate*/,
            auto /*analysis*/
        ) {
            auto changed = change_result::NoChange;
            auto val_pt = lookup(val);
            auto fptr_trg_rep = sets().find(*val_pt);

            auto has_res = call->getNumResults() > 0;

            if (fptr_trg_rep.is_unknown())
                return set_all_unknown();

            if (fptr_trg_rep.is_dummy()) {
                if (auto dummy_info = get_or_create_fn_info(fptr_trg_rep)) {
                    // join previous call with current call
                    for (const auto &[prev, current] : llvm::zip(dummy_info->operands, call.getArgOperands())) {
                        changed |= join_var(prev, *lookup(current));
                    }
                    if (has_res) {
                        changed |= join_var(call->getResult(0), dummy_info->res);
                    }
                } else {
                    auto res_dummy = new_dummy();
                    info->fn_infos.emplace(fptr_trg_rep, fn_info(call.getArgOperands(), res_dummy));
                    if (has_res) {
                        changed |= join_var(call->getResult(0), res_dummy);
                    }
                    changed |= change_result::Change;
                }
            } else if (fptr_trg_rep.is_func()) {
                auto fn_details = get_or_create_fn_info(fptr_trg_rep);
                for (auto arg : fn_details->operands) {
                    for (auto operand : call.getArgOperands()) {
                        auto operand_pt = lookup(operand);
                        changed |= join_var(arg, operand_pt);
                    }
                }
                if (call->getNumResults() > 0) {
                    changed |= join_var(call->getResult(0), fn_details->res);
                }
            }
            return changed;
        }

        change_result add_constant(mlir_value val) { return join_empty(val); }

        fn_info *get_or_create_fn_info(elem_t &elemop);
        change_result make_union(elem_t lhs, elem_t rhs);
        change_result join_var(const elem_t &ptr, const elem_t &new_trg);
        change_result join_var(const elem_t &ptr, const elem_t *new_trg) {
            return join_var(ptr, *new_trg);
        }

        change_result set_all_unknown();
        change_result merge(const steensgaard &rhs);

        change_result join(const mlir_dense_abstract_lattice &rhs) override {
            return this->merge(*static_cast< const steensgaard *>(&rhs));
        };

        void print(llvm::raw_ostream &os) const override;

        static void propagate_members_changed(const elem_t *, auto, auto) { return; }
        static void depend_on_members(const elem_t *, auto) { return; }
        static void add_dependencies(mlir::Operation *op, mlir_dense_dfa< steensgaard > *analysis, ppoint point, auto get_or_create) { return; }

        constexpr static bool propagate_assign() { return false; }
        constexpr static bool propagate_call_arg_zip() { return false; }

        alias_res alias(auto lhs, auto rhs) {
            auto lhs_trg = sets().find(*lookup(lhs));
            auto rhs_trg = sets().find(*lookup(rhs));
            if (lhs_trg.is_dummy() || rhs_trg.is_dummy()) {
                return alias_kind::NoAlias;
            }
            if (lhs_trg != rhs_trg) {
                return alias_kind::NoAlias;
            }
            for (auto child : sets().children[lhs_trg]) {
                if (!child.is_dummy() && child != lhs_trg) {
                    return alias_kind::MayAlias;
                }
            }
            for (auto child : sets().children[rhs_trg]) {
                if (!child.is_dummy() && child != rhs_trg) {
                    return alias_kind::MayAlias;
                }
            }
            if (lhs_trg.is_alloca()) {
                return alias_kind::MayAlias;
            }
            return alias_kind::MustAlias;
        };
    };
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const steensgaard &l);
} //namespace potato::analysis
