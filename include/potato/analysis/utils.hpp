#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/Hashing.h>
#include "llvm/ADT/DenseMapInfoVariant.h"
#include <llvm/ADT/StringRef.h>

#include <mlir/IR/Value.h>
POTATO_UNRELAX_WARNINGS

#include "potato/util/common.hpp"

#include <variant>

namespace potato::util {
    template< typename T >
    concept printable = requires(T a, llvm::raw_ostream &os) { a.print(os); };
}

namespace potato::analysis {
    template < typename value_t, template< typename >typename set_t >
    using pt_map = llvm::DenseMap< value_t, set_t< value_t > >;

    // Allows using DenseSet with above template without having to specify default args
    template< typename value_t >
    using dense_set = llvm::DenseSet< value_t >;

    struct pt_element
    {
        enum class elem_kind {
            alloca,
            var,
            func,
            global
        } kind;
        mlir_value val;
        mlir_operation *operation;

        pt_element(elem_kind kind, mlir_value val, mlir_operation *op)
            : kind(kind), val(val), operation(op) {};
        pt_element(mlir_value val) : kind(elem_kind::var), val(val), operation(nullptr) {}

        static pt_element make_func(mlir_operation *op) {
            return pt_element(elem_kind::func, {}, op);
        }

        static pt_element make_glob(mlir_operation *op) {
            return pt_element(elem_kind::global, {}, op);
        }

        static pt_element make_alloca(mlir_operation *op, mlir_value val = {}) {
            return pt_element(elem_kind::alloca, val, op);
        }

        static pt_element make_var(mlir_value val) {
            return pt_element(elem_kind::var, val, nullptr);
        }

        bool is_alloca() const { return kind == elem_kind::alloca; }
        bool is_var() const { return kind == elem_kind::var; }
        bool is_func() const { return kind == elem_kind::func; }
        bool is_global() const { return kind == elem_kind::global; }

        bool operator==(const pt_element &rhs) const = default;

        void print(llvm::raw_ostream &os) const {
            switch (kind) {
                case elem_kind::alloca:
                    os << "mem_alloc" << operation->getLoc();
                    if (val) {
                        os << " for: " << val;
                    }
                    break;
                case elem_kind::var:
                    os << "var:" << val;
                break;
                case elem_kind::func:
                case elem_kind::global:
                    auto symbol = mlir::cast< mlir::SymbolOpInterface >(operation);
                    os << symbol.getName();
                break;
            }
        };
    };

    bool sets_intersect(const auto &lhs, const auto &rhs) {
        for (const auto &lhs_elem : lhs) {
            if (rhs.contains(lhs_elem))
               return true;
        }
        return false;
    }
} // namespace potato::analysis

template<>
struct std::hash< mlir_value > {
    std::size_t operator() (const mlir_value &value) const {
        return mlir::hash_value(value);
    }
};

template<>
struct std::hash< llvm::StringRef > {
    std::size_t operator() (const llvm::StringRef &value) const {
        return llvm::hash_value(value);
    }
};

template <>
struct std::hash< potato::analysis::pt_element > {
    using pt_element = potato::analysis::pt_element;
    std::size_t operator() (const pt_element &value) const {
        return llvm::hash_combine(
            llvm::hash_value(value.kind),
            std::hash< mlir_value >{}(value.val),
            std::hash< mlir_operation * >{}(value.operation)
        );
    }
};

namespace llvm {
    using potato::analysis::pt_element;

    inline raw_ostream &operator <<(raw_ostream &os, const pt_element &value) {
        value.print(os);
        return os;
    }

    template<>
    struct DenseMapInfo< pt_element > {
        using kind_info = DenseMapInfo< pt_element::elem_kind >;
        using val_info = DenseMapInfo< mlir_value >;
        using op_info = DenseMapInfo< mlir_operation * >;
        static inline pt_element getEmptyKey() {
            return pt_element(
                    kind_info::getEmptyKey(),
                    val_info::getEmptyKey(),
                    op_info::getEmptyKey()
            );
        }

        static inline pt_element getTombstoneKey() {
            return pt_element(
                    kind_info::getTombstoneKey(),
                    val_info::getTombstoneKey(),
                    op_info::getTombstoneKey()
            );
        }

        static unsigned getHashValue(const pt_element &val) {
            return llvm::hash_combine(
                    kind_info::getHashValue(val.kind),
                    val_info::getHashValue(val.val),
                    op_info::getHashValue(val.operation)
            );
        }

        static bool isEqual(const pt_element &lhs, const pt_element &rhs) {
            return lhs == rhs;
        }
    };

} // namespace llvm
