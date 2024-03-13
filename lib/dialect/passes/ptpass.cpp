#include "potato/dialect/potato/passes.hpp"
#include "potato/util/warnings.hpp"
#include "potato/dialect/potato/analysis/pt.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
POTATO_UNRELAX_WARNINGS

#include <memory>

#include "passes_details.hpp"

namespace potato::pt
{
    struct PointsToPass : PointsToPassBase< PointsToPass >
    {
        void runOnOperation() override
        {
            auto root = getOperation();
            auto solver = mlir::DataFlowSolver();

            // Load dependencies
            solver.load< mlir::dataflow::SparseConstantPropagation >();
            solver.load< mlir::dataflow::DeadCodeAnalysis >();

            // Load our analysis
            solver.load< analysis::pt_analysis >();

            if (failed(solver.initializeAndRun(root)))
                signalPassFailure();
            analysis::print_analysis_result(solver, root, llvm::outs());
        }
    };

    std::unique_ptr< mlir::Pass > createPointsToPass()
    {
        return std::make_unique< PointsToPass >();
    }
} // namespace potato::pt

