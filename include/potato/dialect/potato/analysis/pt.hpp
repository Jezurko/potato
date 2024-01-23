#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Interfaces/CallInterfaces.h>
POTATO_UNRELAX_WARNINGS

namespace potato::analysis {

struct PointsToLattice : mlir::dataflow::AbstractDenseLattice {
    mlir::ChangeResult join(const mlir::dataflow::AbstractDenseLattice &rhs) override;

    mlir::ChangeResult meet(const mlir::dataflow::AbstractDenseLattice &rhs) override;
};

struct PointsToAnalaysis : mlir::dataflow::DenseDataFlowAnalysis< PointsToLattice > {

    void visitOperation(mlir::Operation *op, const PointsToLattice &before, PointsToLattice *after) override;

    void visitCallControlFlowTransfer(mlir::CallOpInterface call,
                                      mlir::dataflow::CallControlFlowAction action,
                                      const PointsToLattice &before,
                                      PointsToLattice *after) override;

    void visitRegionBranchControlFlowTransfer(mlir::RegionBranchOpInterface branch,
                                              std::optional< unsigned > regionFrom,
                                              std::optional< unsigned > regionTo,
                                              const PointsToLattice &before,
                                              PointsToLattice *after) override;
};

} // potato::analysis
