#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/Hashing.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/utils.hpp"
#include "potato/util/common.hpp"

#include <unordered_map>
#include <unordered_set>

namespace potato::analysis {
//
//    struct stg_elem {
//        std::optional< pt_element > elem;
//        std::optional< size_t > dummy_id;
//
//        stg_elem(mlir_value val)             : elem(pt_element(val)), dummy_id(std::nullopt) {}
//        stg_elem(size_t id)                  : elem(std::nullopt), dummy_id(id) {}
//        stg_elem()                           : elem(std::nullopt), dummy_id(std::nullopt) {}
//
//        stg_elem(pt_element::elem_kind kind, mlir_value val, mlir_operation *op)
//            : elem(pt_element(kind, val, op)), dummy_id(std::nullopt) {}
//
//        static stg_elem make_func(mlir_operation *op) {
//            return stg_elem(pt_element::elem_kind::func, {}, op);
//        }
//
//        static stg_elem make_named_var(mlir_operation *op) {
//            return stg_elem(pt_element::elem_kind::named_var, {}, op);
//        }
//
//        static stg_elem make_alloca(mlir_operation *op, mlir_value val = {}) {
//            return stg_elem(pt_element::elem_kind::alloca, val, op);
//        }
//
//        static stg_elem make_var(mlir_value val) {
//            return stg_elem(pt_element::elem_kind::var, val, nullptr);
//        }
//
//        bool operator==(const stg_elem &rhs) const = default;
//
//        inline bool is_unknown() const { return !is_dummy() && !elem.has_value(); }
//        inline bool is_dummy() const { return dummy_id.has_value(); }
//        inline bool is_top() const { return is_unknown(); }
//        inline bool is_bottom() const { return is_dummy(); }
//        inline bool is_alloca() const { return elem && elem->is_alloca(); }
//        inline bool is_var() const { return elem && elem->is_var(); }
//        inline bool is_func() const { return elem && elem->is_func(); }
//        inline bool is_named_var() const { return elem && elem->is_named_var(); }
//
//        void print(llvm::raw_ostream &os) const;
//    };
//
//    struct fn_info {
//        std::vector< stg_elem > operands;
//
//        stg_elem res;
//        bool operator==(const fn_info &rhs) const = default;
//    };
//
//
//    llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const stg_elem &e);
//} // namespace potato::analysis
//
//template <>
//struct std::hash< potato::analysis::stg_elem > {
//    using stg_elem = potato::analysis::stg_elem;
//    using pt_element = potato::analysis::pt_element;
//
//    std::size_t operator() (const stg_elem &value) const {
//        return llvm::hash_combine(std::hash< std::optional< pt_element > >{}(value.elem),
//                                  std::hash< std::optional< size_t > >{}(value.dummy_id));
//    }
//};
//
//namespace potato::analysis {
//    namespace detail {
//        template< typename T >
//        struct union_find {
//            std::unordered_map< T, T > parents;
//            std::unordered_map< T, size_t > rank;
//
//            std::unordered_map< T, std::unordered_set< T > > children;
//
//            bool insert(const T& elem) {
//                if (parents.find(elem) == parents.end()) {
//                    parents[elem] = elem;
//                    rank[elem] = 0;
//                    children[elem] = {elem};
//                    return true;
//                }
//                return false;
//            };
//
//            T find(const T &x) {
//                auto find_it = parents.find(x);
//                if (find_it == parents.end()) {
//                    insert(x);
//                    find_it = parents.find(x);
//                }
//
//                T found = x;
//                T parent = find_it->second;
//                while (found != parents[found]) {
//                    auto &parent_parent = parents[parent];
//                    parents[found] = parent_parent;
//                    found = parent_parent;
//                    parent = parents[found];
//                }
//                return found;
//            }
//
//            T set_union(const T &x, const T &y) {
//                T x_root = find(x);
//                T y_root = find(y);
//                if (x_root == y_root)
//                    return x_root;
//
//                size_t &x_rank = rank[x_root];
//                size_t &y_rank = rank[y_root];
//
//                if (x_root.is_unknown()) {
//                    parents[y_root] = x_root;
//                    children[x_root].insert(y_root);
//                    if (y_rank >= x_rank) {
//                        x_rank = y_rank + 1;
//                    }
//                    return x_root;
//                }
//                if (y_root.is_unknown()) {
//                    parents[x_root] = y_root;
//                    children[y_root].insert(x_root);
//                    if (x_rank >= y_rank) {
//                        y_rank = x_rank + 1;
//                    }
//                    return y_root;
//                }
//
//                if (x_root.is_dummy() || y_root.is_func()) {
//                    parents[x_root] = y_root;
//                    children[y_root].insert(x_root);
//                    if (x_rank >= y_rank) {
//                        y_rank = x_rank + 1;
//                    }
//                    return y_root;
//                }
//                if (y_root.is_dummy() || x_root.is_func()) {
//                    parents[y_root] = x_root;
//                    children[x_root].insert(y_root);
//                    if (y_rank >= x_rank) {
//                        x_rank = y_rank + 1;
//                    }
//                    return x_root;
//                }
//
//                if (x_rank > y_rank) {
//                    parents[y_root] = x_root;
//                    children[x_root].insert(y_root);
//                    return x_root;
//                } else {
//                    parents[x_root] = y_root;
//                    children[y_root].insert(x_root);
//                    if (x_rank == y_rank) {
//                        y_rank++;
//                    }
//                    return y_root;
//                }
//            }
//            // TODO: children iterator wrapper
//        };
//    } // namespace detail

    //struct steensgaard_lattice : pt_lattice {
    //};

    //struct steensgaard : pt_analysis< steensgaard, steensgaard_lattice > {
    //};
    struct steensgaard {
    };
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const steensgaard &l);
} //namespace potato::analysis
