#include "potato/analysis/andersen.hpp"
#include "potato/passes/analysis/analysis.hpp"
#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
POTATO_UNRELAX_WARNINGS

#include <memory>

namespace potato::pt {
#define GEN_PASS_DEF_AABENCHPASS
#include "potato/passes/analysis/Analysis.h.inc"

    struct AABenchPass : impl::AABenchPassBase< AABenchPass > {
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
            //auto analysis = util::get_analysis< analysis::aa_lattice >(solver, root);
            //if (analysis->is_all_unknown()) {
            //    signalPassFailure();
            //}
        }
    };

    std::unique_ptr< mlir::Pass > createAABenchPass() {
        return std::make_unique< AABenchPass >();
    }
} // namespace potato::pt

