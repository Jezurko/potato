#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/IR/Value.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SetVector.h>
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/potato/potato.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis {

struct pt_lattice : mlir_dense_abstract_lattice
{
    pt_map pt_relation;

    change_result merge(const pt_lattice &rhs) {
        change_result res = change_result::NoChange;
        for (const auto &[key, rhs_value] : rhs.pt_relation) {
            auto &lhs_value = pt_relation[key];
            if (lhs_value.set_union(rhs_value)) {
                res |= change_result::Change;
            }
        }
        return res;
    }

    change_result join(const mlir_dense_abstract_lattice &rhs) override {
        return this->merge(*static_cast< const pt_lattice *>(&rhs));
    };

    mlir::ChangeResult meet(const mlir_dense_abstract_lattice &rhs) override;
};

struct pt_analysis : mlir_dense_dfa< pt_lattice >
{

    void visit_pt_op(const pt::AddressOfOp &op, const pt_lattice &before, pt_lattice *after);

    void visit_pt_op(const pt::CopyOp &op, const pt_lattice &before, pt_lattice *after);

    void visit_pt_op(const pt::AssignOp &op, const pt_lattice &before, pt_lattice *after);

    void visit_pt_op(const pt::DereferenceOp &op, const pt_lattice &before, pt_lattice *after);

    void visit_pt_op(const pt::MAllocOp &op, const pt_lattice &before, pt_lattice *after);

    void visitOperation(mlir::Operation *op, const pt_lattice &before, pt_lattice *after) override;

    void visitCallControlFlowTransfer(mlir::CallOpInterface call,
                                      mlir::dataflow::CallControlFlowAction action,
                                      const pt_lattice &before,
                                      pt_lattice *after) override;

    void visitRegionBranchControlFlowTransfer(mlir::RegionBranchOpInterface branch,
                                              std::optional< unsigned > regionFrom,
                                              std::optional< unsigned > regionTo,
                                              const pt_lattice &before,
                                              pt_lattice *after) override;
};

} // potato::analysis
