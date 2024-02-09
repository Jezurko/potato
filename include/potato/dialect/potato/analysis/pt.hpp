#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SetVector.h>
POTATO_UNRELAX_WARNINGS

namespace potato::analysis {

using pt_map = llvm::DenseMap< mlir::Value, llvm::SetVector< mlir::Value > >;
using change_result = mlir::ChangeResult;

struct PointsToLattice : mlir::dataflow::AbstractDenseLattice
{
    pt_map pt_relation;

    change_result merge(const PointsToLattice &rhs) {
        change_result res = change_result::NoChange;
        for (auto &[key, rhs_value] : rhs.pt_relation) {
            auto &lhs_value = pt_relation[key];
            if (lhs_value.set_union(rhs_value)) {
                res |= change_result::Change;
            }
        }
        return res;
    }

    change_result join(const mlir::dataflow::AbstractDenseLattice &rhs) override {
        return this->merge(*static_cast< const PointsToLattice *>(&rhs));
    };

    mlir::ChangeResult meet(const mlir::dataflow::AbstractDenseLattice &rhs) override;
};

struct PointsToAnalaysis : mlir::dataflow::DenseDataFlowAnalysis< PointsToLattice >
{

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
