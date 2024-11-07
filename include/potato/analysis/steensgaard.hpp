#pragma once

#include "potato/analysis/lattice.hpp"
#include "potato/analysis/utils.hpp"
#include "potato/dialect/ops.hpp"
#include "potato/util/common.hpp"

#include <memory>
#include <unordered_map>
#include <vector>
#include <string>

namespace potato::analysis {
    namespace detail {
        template< typename T >
        struct union_find {
            std::unordered_map< T, T > parents;
            std::unordered_map< T, size_t > rank;

            std::unordered_map< T, std::set< T > > children;

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
                }

                T found = x;
                const T &parent = find_it->second;
                while (found != parent) {
                    auto &parent_parent = parents[parent];
                    parents[found] = parent_parent;
                    found = parent_parent;
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

    struct stg_elem {
        std::optional< pt_element > elem;

        stg_elem(const llvm::StringRef name) : elem(pt_element(name)) {}
        stg_elem(mlir_value val)             : elem(pt_element(val)) {}
        stg_elem() : elem(std::nullopt) {}

        bool operator==(const stg_elem &rhs) const = default;

        bool is_unknown() {
            return elem.has_value();
        }
    };

    struct steensgaard : mlir_dense_abstract_lattice {
        using mlir_dense_abstract_lattice::AbstractDenseLattice;
        using elem_t = stg_elem;
        std::shared_ptr< detail::union_find< elem_t > > sets;
        // edges between nodes(sets)
        // keep the invariant, that keys are the set roots to minimize the number of edges
        // targets can be any set members
        std::shared_ptr< std::unordered_map< elem_t, elem_t > > mapping;

        // static bool all_unknown;

        static unsigned int mem_loc_count;
        unsigned int alloc_count();

        std::optional< std::string > alloc_name = {};
        llvm::StringRef get_alloc_name();

        elem_t lookup(const elem_t &val) const {
            return sets->find(val);
        }

        static auto new_symbol(const llvm::StringRef name) {
            return elem_t(name);
        }

        static auto new_top_set() {
            // default constructor creates unknown
            return elem_t();
        }

        auto join_empty(mlir_value val) {
            sets->insert(val);
        }

        auto add_constant(mlir_value val) {
            return join_empty(val);
        }

        change_result make_union(elem_t lhs, elem_t rhs) {
            auto lhs_root = sets->find(lhs);
            auto rhs_root = sets->find(rhs);
            if (lhs_root == rhs_root) {
                return change_result::NoChange;
            }

            auto lhs_trg = mapping->find(lhs_root);
            auto rhs_trg = mapping->find(rhs_root);

            auto new_root = sets->set_union(lhs, rhs);

            if (lhs_trg == mapping->end() && rhs_trg == mapping->end()) {
                return change_result::Change;
            }

            // merge outgoing edges
            if (lhs_trg == mapping->end() && rhs_root != new_root) {
                mapping->insert({new_root, rhs_trg->second});
                mapping->erase(rhs_trg);
                return change_result::Change;
            }
            if (rhs_trg == mapping->end() && lhs_root != new_root) {
                mapping->insert({new_root, lhs_trg->second});
                mapping->erase(lhs_trg);
                return change_result::Change;
            }

            if (lhs_trg->second.is_unknown()) {
                (*mapping)[rhs_root] = lhs_trg->second;
                return change_result::Change;
            }

            if (rhs_trg->second.is_unknown()) {
                (*mapping)[lhs_root] = rhs_trg->second;
            }

            // if both have outgoing edge, unify the targets and remove the redundant edge
            std::ignore = make_union(lhs_trg->second, rhs_trg->second);
            if (new_root == lhs_root) {
                mapping->erase(rhs_trg);
            } else {
                mapping->erase(lhs_trg);
            }

            return change_result::Change;
        }

        change_result join_var(const elem_t &ptr, const elem_t &new_trg) {
            auto ptr_rep = sets->find(ptr);
            auto new_trg_rep = sets->find(new_trg);

            auto ptr_trg = mapping->find(ptr_rep);
            if (ptr_trg == mapping->end()) {
                mapping->insert({ptr_rep, new_trg_rep});
                return change_result::Change;
            }
            if (ptr_trg->second.is_unknown()) {
                return change_result::NoChange;
            }
            if (new_trg_rep.is_unknown()) {
                (*mapping)[ptr_rep] = new_trg_rep;
                return change_result::Change;
            }
            return make_union(ptr_trg->second, new_trg_rep);
        }

        change_result set_all_unknown();

        // store map elem->target
        // store UF of all elems that point to the same
        // %x = copy %y - x.trg = find(y.trg)
        // %x = addr_of glob - x.trg = find(glob)
        // %x = *y - %x x.trg = find(find(y.trg).trg)
        // %x = alloc - %x.trg = alloc; insert(alloc)
        // assign %y to %x = union(find(x.trg), find(y))

        change_result merge(const steensgaard &rhs);
        change_result intersect(const steensgaard &rhs);

        change_result join(const mlir_dense_abstract_lattice &rhs) override {
            return this->merge(*static_cast< const steensgaard *>(&rhs));
        };

        change_result meet(const mlir_dense_abstract_lattice &rhs) override {
            return this->intersect(*static_cast< const steensgaard *>(&rhs));
        };

        void print(llvm::raw_ostream &os) const override;

        static void add_dependencies(mlir::Operation *op, mlir_dense_dfa< steensgaard > *analysis, ppoint point, auto get_or_create);
        alias_res alias(auto lhs, auto rhs) const;
    };
} //namespace potato::analysis
