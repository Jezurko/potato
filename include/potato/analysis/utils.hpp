#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/Hashing.h>
POTATO_UNRELAX_WARNINGS

#include "potato/util/common.hpp"

#include <string>

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
        mlir_value val;
        std::string name;

        bool operator==(const pt_element &rhs) const {
            if (val || rhs.val)
                return val == rhs.val;
            return name == rhs.name;
        }

        void print(llvm::raw_ostream &os) const { os << name << ": " << val; };
    };

    bool sets_intersect(const auto &lhs, const auto &rhs) {
        for (const auto &lhs_elem : lhs) {
            if (rhs.contains(lhs_elem))
               return true;
        }
        return false;
    }
} // namespace potato::analysis


namespace llvm {
    using potato::analysis::pt_element;

    inline raw_ostream &operator <<(raw_ostream &os, const pt_element &value) {
        value.print(os);
        return os;
    }

    template<>
    struct DenseMapInfo< pt_element > {
        using value_info = DenseMapInfo< mlir_value >;
        static inline pt_element getEmptyKey() {
            return {value_info::getEmptyKey(), ""};
        }

        static inline pt_element getTombstoneKey() {
            return {value_info::getTombstoneKey(), ""};
        }

        static unsigned getHashValue(const pt_element &val) {
            return val.val ? value_info::getHashValue(val.val) : (unsigned) llvm::hash_value(val.name);
        }

        static bool isEqual(const pt_element &lhs, const pt_element &rhs) {
            return lhs == rhs;
        }
    };

} // namespace llvm
