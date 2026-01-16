#include "potato/passes/analysis/analysis.hpp"
#include "potato/util/test.hpp"
#include "potato/util/warnings.hpp"
#include "potato/analysis/andersen.hpp"
#include "potato/analysis/pt.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
POTATO_UNRELAX_WARNINGS

#include <memory>

namespace potato::pt {
#define GEN_PASS_DEF_POINTSTOPASS
#include "potato/passes/analysis/Analysis.h.inc"

    struct PointsToPass : impl::PointsToPassBase< PointsToPass > {
        void runOnOperation() override {
            auto root = getOperation();
            auto solver = mlir::DataFlowSolver();

            // Load dependencies
            solver.load< mlir::dataflow::SparseConstantPropagation >();
            solver.load< mlir::dataflow::DeadCodeAnalysis >();

            // Load our analysis
            solver.load< analysis::aa_analysis >();

            if (failed(solver.initializeAndRun(root)))
                signalPassFailure();

            if (print_lattice)
                analysis::aa_analysis::print_analysis(root, solver, llvm::outs());
            test::check_aliases< analysis::aa_lattice >(solver, root);
            //if (print_stats)
            //    analysis::print_analysis_stats(solver, root, llvm::outs());
            //if (print_func_stats)
            //    analysis::print_analysis_func_stats(solver, root, llvm::outs());
        }
    };

    std::unique_ptr< mlir::Pass > createPointsToPass() {
        return std::make_unique< PointsToPass >();
    }
} // namespace potato::pt

