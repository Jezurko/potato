#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Analysis/CallGraph.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
POTATO_UNRELAX_WARNINGS

#include "potato/util/common.hpp"

namespace potato::test {
    constexpr auto may_alias  = "may_alias";
    constexpr auto must_alias = "must_alias";
    constexpr auto no_alias   = "no_alias";
    template< typename analysis_lattice >
    void check_aliases(mlir::DataFlowSolver &solver, mlir_operation *root) {
        auto lattice = util::get_analysis< analysis_lattice >(solver, root);
        root->walk([&](mlir::CallOpInterface call) {
            auto fn = mlir::dyn_cast_if_present< mlir::FunctionOpInterface >(call.resolveCallable());
            if (!fn) {
                return;
            }
            auto fn_name = fn.getName();

            alias_kind kind;
            if (fn_name == may_alias) {
                kind = alias_kind::MayAlias;
            } else if (fn_name == no_alias) {
                kind = alias_kind::NoAlias;
            } else if (fn_name == must_alias) {
                kind = alias_kind::MustAlias;
            } else {
                return;
            }

            auto expected_res = alias_res(kind);
            size_t args_c = call->getNumOperands();
            for (size_t i = 0; i < args_c; i++) {
                for (size_t j = i + 1; j < args_c; j++) {
                    auto alias = lattice->alias(call->getOperand(i), call->getOperand(j));
                    if (alias != expected_res) {
                        llvm::errs()
                            << "Arguments " << i << " and " << j
                            << " of call at " << call.getLoc()
                            << " do not " << expected_res << "\n";
                        assert(false);
                    }
                }
            }
        });
    }
}
