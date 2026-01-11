#include "potato/passes/analysis/analysis.hpp"
#include "potato/util/warnings.hpp"
#include "potato/analysis/trad/llvm_aa.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
POTATO_UNRELAX_WARNINGS

#include <memory>

namespace potato::pt {
#define GEN_PASS_DEF_LLVMPOINTSTOPASS
#include "potato/passes/analysis/Analysis.h.inc"

    struct LLVMPointsToPass : impl::LLVMPointsToPassBase< LLVMPointsToPass > {
        void runOnOperation() override {
            auto root = getOperation();
            auto solver = mlir::DataFlowSolver();

            // Load dependencies
            solver.load< mlir::dataflow::SparseConstantPropagation >();
            solver.load< mlir::dataflow::DeadCodeAnalysis >();

            // Load our analysis
            solver.load< analysis::trad::llvm_andersen >();

            if (failed(solver.initializeAndRun(root)))
                signalPassFailure();

            if (print_lattice)
                analysis::trad::print_analysis_result(solver, root, llvm::outs());
            //if (print_stats)
            //    analysis::trad::print_analysis_stats(solver, root, llvm::outs());
            //if (print_func_stats)
            //    analysis::trad::print_analysis_func_stats(solver, root, llvm::outs());
        }
    };

    std::unique_ptr< mlir::Pass > createLLVMPointsToPass() {
        return std::make_unique< LLVMPointsToPass >();
    }
} // namespace potato::pt

