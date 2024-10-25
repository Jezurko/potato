#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/CallGraph.h>

#include <llvm/ADT/Hashing.h>
POTATO_UNRELAX_WARNINGS

#include "potato/util/common.hpp"

#include <ranges>
#include <unordered_map>
#include <deque>

template <>
struct std::hash< std::deque< mlir_operation *  > > {
    std::size_t operator() (const std::deque< mlir_operation * > &value) const {
        const auto hash_range = value | std::views::transform([](mlir_operation * op) { return llvm::hash_value(op); });
        return llvm::hash_combine_range(hash_range.begin(), hash_range.end());
    }
};

namespace potato::analysis {
    template< typename lattice, unsigned context_size >
    struct call_context_wrapper : mlir_dense_abstract_lattice {
        using mlir_dense_abstract_lattice::AbstractDenseLattice;

        call_context_wrapper(ppoint point) : AbstractDenseLattice(point) {
            // Always init with no context
            ctx_lattice.insert({ context_t(), { lattice(), change_result::Change } });
        }

        change_result join(const mlir_dense_abstract_lattice &rhs) override {
            const auto &wrapped = *static_cast< const call_context_wrapper * >(&rhs);
            auto changed = change_result::NoChange;
            for (const auto &[ctx, lattice_with_cr] : wrapped) {
                auto [lhs_with_cr, inserted] = add_context(ctx, lattice_with_cr.first);
                if (inserted) {
                    changed |= change_result::Change;
                } else {
                    auto &[lhs_lattice, changed_lhs] = *lhs_with_cr;
                    changed_lhs = lhs_lattice.join(lattice_with_cr.first);
                    changed |= changed_lhs;
                }
            }
            return changed;
        };

        change_result meet(const mlir_dense_abstract_lattice &rhs) override {
            auto &wrapped = *static_cast< const call_context_wrapper * >(&rhs);
            auto changed = change_result::NoChange;
            for (const auto &[ctx, lattice_with_cr] : wrapped) {
                // Should really meet *add* stuff?
                auto [lhs_with_cr, inserted] = add_context(ctx, lattice_with_cr.first);
                if (inserted) {
                    changed |= change_result::Change;
                } else {
                    auto &[lhs_lattice, changed_lhs] = *lhs_with_cr;
                    changed_lhs = lhs_lattice.meet(lattice_with_cr.first);
                    changed |= changed_lhs;
                }
            }
            return changed;
        };

        using context_t = std::deque< mlir_operation * >;
        using ctx_map = std::unordered_map< context_t, std::pair< lattice, change_result > >;
        using lattice_change_pair = std::pair< lattice, change_result >;

        using iterator       = typename ctx_map::iterator;
        using const_iterator = typename ctx_map::const_iterator;
        iterator begin() { return ctx_lattice.begin(); }
        iterator end() { return ctx_lattice.end(); }
        const_iterator begin() const { return ctx_lattice.begin(); }
        const_iterator end() const { return ctx_lattice.end(); }

        std::pair< lattice_change_pair *, bool > add_context(context_t ctx, const lattice &state) {
            if (ctx.size() > context_size)
                ctx.pop_front();
            auto [value_it, inserted] = ctx_lattice.insert({std::move(ctx), {state, change_result::Change}});
            return {&value_it->second, inserted};
        }

        lattice_change_pair *get_for_context(const context_t &context) {
            auto lattice_it = ctx_lattice.find(context);
            return lattice_it == ctx_lattice.end() ? nullptr : &lattice_it->second;
        };

        const lattice_change_pair *get_for_context(const context_t &context) const {
            auto lattice_it = ctx_lattice.find(context);
            return lattice_it == ctx_lattice.end() ? nullptr : &lattice_it->second;
        }

        lattice_change_pair &get_for_default_context() { return *get_for_context({}); }
        const lattice_change_pair &get_for_default_context() const { return *get_for_context({}); }

        void print(llvm::raw_ostream &/*os*/) const override {
            return;
        }

        private:
            ctx_map ctx_lattice;
    };

} // namespace potato::analysis
