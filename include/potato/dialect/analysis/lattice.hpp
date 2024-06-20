#pragma once
#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/SetOperations.h>
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/analysis/utils.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis {

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

        change_result join(const lattice_set< value_t > &rhs) {
            if (rhs.is_top()) {
                if (!is_top()) {
                    state = state::top;
                    set.clear();
                    return change_result::Change;
                }
                return change_result::NoChange;
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

        private:
        lattice_set(enum state s) : state(s) {}
    };
} // namespace potato::analysis
