
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
struct std::hash< std::vector< cg_edge > > {
    std::size_t operator() (const std::vector< cg_edge > &value) const {
        auto hashing = [](const cg_edge &edge) {
            auto kind = edge.isAbstract() ? 0 : edge.isCall() ? 1 : 2;
            auto trg_and_kind = llvm::PointerIntPair< cg_node *, 2>(edge.getTarget(), kind);
            return llvm::DenseMapInfo< decltype(trg_and_kind) >::getHashValue(trg_and_kind);
        };
        const auto hash_range = value | std::views::transform(hashing);
        return llvm::hash_combine_range(hash_range.begin(), hash_range.end());
    }
};

namespace potato::analysis {
    template< typename lattice >
    struct call_context_wrapper {
        using context_t = std::vector< cg_edge >;
        using ctx_map = std::unordered_map< context_t, std::pair< lattice, change_result > >;
        using lattice_change_pair = std::pair< lattice, change_result >;

        using iterator       = typename ctx_map::iterator;
        using const_iterator = typename ctx_map::const_iterator;
        iterator begin() { return ctx_lattice.begin(); }
        iterator end() { return ctx_lattice.end(); }
        const_iterator begin() const { return ctx_lattice.begin(); }
        const_iterator end() const { return ctx_lattice.end(); }

        std::pair< lattice, change_result > &add_context(const context_t &ctx_prefix, const cg_edge &last, const lattice &state) {
            const auto it = ctx_lattice.find(ctx_prefix);
            if (it != ctx_lattice.end()) {
                for (const auto &edge : it->first) {
                    if (edge == last)
                        return it->second;
                }
            }
            auto ctx = ctx_prefix;
            ctx.push_back(last);
            auto [value_it, inserted] = ctx_lattice.insert({ ctx, {state, change_result::Change } });
            return value_it->second;
        };

        lattice_change_pair *get_for_context(const context_t &context) {
            auto lattice_it = ctx_lattice.find(context);
            return lattice_it == ctx_lattice.end() ? nullptr : &lattice_it->second;
        };

        const std::pair< lattice, change_result > *get_for_context(const context_t &context) const {
            auto lattice_it = ctx_lattice.find(context);
            return lattice_it == ctx_lattice.end() ? nullptr : &lattice_it->second;
        }

        lattice_change_pair *get_for_default_context() { return get_for_context({}); }
        const lattice_change_pair *get_for_default_context() const { return get_for_context({}); }

        private:
            ctx_map ctx_lattice;
    };

} // namespace potato::analysis
