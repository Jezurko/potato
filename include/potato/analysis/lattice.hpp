#pragma once
#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/SetOperations.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/utils.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis {
    template< typename T >
    concept printable = requires(T a, llvm::raw_ostream &os) { a.print(os); };

    template< typename value_t >
    struct lattice_set {
        enum class state {
            concrete,
            top
        } state;

        dense_set< value_t > set;

        lattice_set() : state(state::concrete) {}
        lattice_set(const dense_set< value_t > &set) : state(state::concrete), set(set) {}
        lattice_set(value_t element) : state(state::concrete) { set.insert(element); }

        bool is_top() const { return state == state::top; }
        bool is_concrete() const { return state == state::concrete && !set.empty(); }
        bool is_bottom() const { return state == state::concrete && set.empty(); }
        bool is_single_target() const { return state == state::concrete && set.size() == 1; }

        bool operator==(const lattice_set< value_t > &rhs) const {
            if (state != rhs.state) { return false; }
            return set == rhs.set;
        }

        change_result join(const lattice_set< value_t > &rhs) {
            if (is_top()) {
                return change_result::NoChange;
            }
            if (rhs.is_top()) {
                state = state::top;
                set.clear();
                return change_result::Change;
            }
            if (llvm::set_union(set, rhs.get_set_ref())) {
                return change_result::Change;
            }
            return change_result::NoChange;
        };

        change_result meet(const lattice_set< value_t > &rhs) {
            if (rhs.is_top()) {
                return change_result::NoChange;
            }
            if (is_top()) {
                state = state::concrete;
                set = rhs.get_set_ref();
                return change_result::Change;
            }
            if (util::set_intersect(set, rhs.get_set_ref())) {
                return change_result::Change;
            }
            return change_result::NoChange;
        }

        auto insert(value_t &&element) {
            return set.insert(element);
        }

        auto insert(const value_t &element) {
            return set.insert(element);
        }

        change_result clear() {
            if (is_bottom()) {
                return change_result::NoChange;
            }
            set.clear();
            state = state::concrete;
            return change_result::Change;
        }

        void subtract(const lattice_set< value_t > &rhs) {
            llvm::set_subtract(set, rhs.get_set_ref());
        }

        void subtract(const dense_set< value_t > &rhs) {
            llvm::set_subtract(set, rhs);
        }

        dense_set< value_t > get_set() const { return set; }
        dense_set< value_t > &get_set_ref() { return set; }
        const dense_set< value_t > &get_set_ref() const { return set; }

        static lattice_set< value_t > make_top() { return { state::top }; }

        void print(llvm::raw_ostream &os) const
        requires printable< value_t >
        {
            if (is_top()) {
                os << "{ TOP }";
                return;
            }
            std::string sep;
            os << "{";
            for (const auto &elem : set) {
                os << sep << elem;
                sep = ", ";
            }
            os << "}";
        }

        void print(llvm::raw_ostream &os) const
        requires (!printable< value_t >)
        {
            if (is_top()) {
                os << "{ TOP }";
                return;
            }
            os << "{ concrete set with " << set.size() << "elements }";
        }

        private:
        lattice_set(enum state s) : state(s) {}
    };
} // namespace potato::analysis

namespace llvm {
    template< typename T >
    inline raw_ostream &operator <<(raw_ostream &os, const potato::analysis::lattice_set< T > &value) {
        value.print(os);
        return os;
    }
} // namespace llvm
