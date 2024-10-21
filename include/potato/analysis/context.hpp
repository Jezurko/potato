#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/CallGraph.h>

#include <llvm/ADT/Hashing.h>
POTATO_UNRELAX_WARNINGS

#include "potato/util/common.hpp"

#include <ranges>
#include <unordered_map>
#include <vector>

template <>
struct std::hash< std::vector< std::pair< cg_node *, cg_edge > > > {
    std::size_t operator() (const std::vector< std::pair< cg_node *, cg_edge > > &value) const {
        auto hashing = [](const std::pair< cg_node *, cg_edge > val) {
            auto kind = val.second.isAbstract() ? 0 : val.second.isCall() ? 1 : 2;
            auto trg_and_kind = llvm::PointerIntPair< cg_node *, 2>(val.second.getTarget(), kind);
            return llvm::hash_combine(llvm::hash_value(val.first), llvm::DenseMapInfo< decltype(trg_and_kind) >::getHashValue(trg_and_kind));
        };
        const auto hash_range = value | std::views::transform(hashing);
        return llvm::hash_combine_range(hash_range.begin(), hash_range.end());
    }
};

namespace potato::analysis {
    template< typename lattice >
    struct call_context_wrapper : mlir_dense_abstract_lattice {
        using mlir_dense_abstract_lattice::AbstractDenseLattice;

        change_result join(const mlir_dense_abstract_lattice &rhs) override {
            auto &wrapped = *static_cast< const call_context_wrapper * >(&rhs);
            auto changed = change_result::NoChange;
            for (const auto &[ctx, lattice_with_cr] : wrapped) {
                if (auto lhs_lattice = get_for_context(ctx)) {
                    changed |= lhs_lattice->join(rhs);
                } else {
                    auto &[new_lhs_lattice, changed_lhs] = propagate_context(ctx, lattice_with_cr->first);
                    changed |= changed_lhs;
                    changed |= new_lhs_lattice.join(rhs);
                }

            }
            return changed;
        };

        change_result meet(const mlir_dense_abstract_lattice &rhs) override {
            auto &wrapped = *static_cast< const call_context_wrapper * >(&rhs);
            auto changed = change_result::NoChange;
            for (const auto &[ctx, lattice_with_cr] : wrapped) {
                if (auto lhs_lattice = get_for_context(ctx)) {
                    changed |= lhs_lattice->meet(rhs);
                } else {
                    // Should really meet *add* stuff?
                    auto &[new_lhs_lattice, changed_lhs] = propagate_context(ctx, lattice_with_cr->first);
                    changed |= changed_lhs;
                    changed |= new_lhs_lattice.meet(rhs);
                }

            }
            return changed;
        };

        using context_t = std::vector< std::pair< cg_node *, cg_edge > >;
        using ctx_map = std::unordered_map< context_t, std::pair< lattice, change_result > >;
        using lattice_change_pair = std::pair< lattice, change_result >;

        using iterator       = typename ctx_map::iterator;
        using const_iterator = typename ctx_map::const_iterator;
        iterator begin() { return ctx_lattice.begin(); }
        iterator end() { return ctx_lattice.end(); }
        const_iterator begin() const { return ctx_lattice.begin(); }
        const_iterator end() const { return ctx_lattice.end(); }

        // Method for adding a new context with the necessary checks
        lattice_change_pair &add_new_context(const context_t &ctx_prefix, const std::pair< cg_node *, cg_edge > &last, const lattice &state) {
            const auto it = ctx_lattice.find(ctx_prefix);
            if (it != ctx_lattice.end()) {
                for (const auto &edge : it->first) {
                    if (edge == last)
                        return it->second;
                }
            }
            auto ctx = ctx_prefix;
            ctx.push_back(last);
            return propagate_context(ctx, state);
        }

        // This method serves to propagate context from a previous (before) lattice
        // It does not do all the necessary checks for inicialization of an unknown context
        lattice_change_pair &propagate_context(const context_t &ctx, const lattice &state) {
            auto [value_it, inserted] = ctx_lattice.insert({ ctx, { state, change_result::Change } });
            return value_it;
        }

        lattice_change_pair *get_for_context(const context_t &context) {
            auto lattice_it = ctx_lattice.find(context);
            return lattice_it == ctx_lattice.end() ? nullptr : &lattice_it->second;
        };

        const lattice_change_pair *get_for_context(const context_t &context) const {
            auto lattice_it = ctx_lattice.find(context);
            return lattice_it == ctx_lattice.end() ? nullptr : &lattice_it->second;
        }

        lattice_change_pair *get_for_default_context() { return get_for_context({}); }
        const lattice_change_pair *get_for_default_context() const { return get_for_context({}); }

        private:
            ctx_map ctx_lattice;
    };

} // namespace potato::analysis
