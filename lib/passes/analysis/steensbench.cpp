#include "potato/analysis/steensgaard.hpp"
#include "potato/analysis/pt.hpp"
#include "potato/passes/analysis.hpp"
#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlowFramework.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h>
POTATO_UNRELAX_WARNINGS

#include <memory>

#include "passes_details.hpp"

namespace potato::pt
{
    struct SteensBenchPass : SteensBenchPassBase< SteensBenchPass >
    {
        void runOnOperation() override
        {
            auto root = getOperation();
            auto solver = mlir::DataFlowSolver();

            // Load dependencies
            solver.load< mlir::dataflow::SparseConstantPropagation >();
            solver.load< mlir::dataflow::DeadCodeAnalysis >();

            // Load our analysis
            solver.load< analysis::pt_analysis< analysis::steensgaard > >();

            if (failed(solver.initializeAndRun(root)))
                signalPassFailure();
            auto analysis = util::get_analysis< analysis::steensgaard >(solver, root);
            if (analysis->is_all_unknown()) {
                signalPassFailure();
            }
        }
    };

    std::unique_ptr< mlir::Pass > createSteensBenchPass()
    {
        return std::make_unique< SteensBenchPass >();
    }
} // namespace potato::pt

