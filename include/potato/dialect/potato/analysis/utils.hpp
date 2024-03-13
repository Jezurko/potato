#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/Hashing.h>
POTATO_UNRELAX_WARNINGS

#include "potato/util/common.hpp"

#include <string>

namespace potato::analysis {

template < typename value_t >
using pt_map = llvm::DenseMap< value_t, llvm::SetVector< value_t > >;

struct pt_element
{
    value val;
    std::string name;

    pt_element(value &&value, std::string &&name) : val(value), name(name) {};
    pt_element(value &&value) : val(value), name("") {};

    bool operator==(const pt_element &rhs) const {
        if (val || rhs.val)
            return val == rhs.val;
        return name == rhs.name;
    }

    void print(llvm::raw_ostream &os) const { os << name << ": " << val; };
};

} // namespace potato::analysis


namespace llvm {
using potato::analysis::pt_element;

inline raw_ostream &operator <<(raw_ostream &os, const pt_element &value) {
    value.print(os);
    return os;
}

template<>
struct DenseMapInfo< pt_element > {
    using value_info = DenseMapInfo< value >;
    static inline pt_element getEmptyKey() {
        return {value_info::getEmptyKey()};
    }

    static inline pt_element getTombstoneKey() {
        return {value_info::getTombstoneKey()};
    }

    static unsigned getHashValue(const pt_element &val) {
        return val.val ? value_info::getHashValue(val.val) : (unsigned) llvm::hash_value(val.name);
    }

    static bool isEqual(const pt_element &lhs, const pt_element &rhs) {
        return lhs == rhs;
    }
};

} // namespace llvm
