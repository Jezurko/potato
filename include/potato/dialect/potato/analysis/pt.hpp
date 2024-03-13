#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/TypeSwitch.h>
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/potato/potato.hpp"
#include "potato/dialect/potato/analysis/utils.hpp"
#include "potato/util/common.hpp"

#include <string>

namespace potato::analysis {

struct pt_lattice : mlir_dense_abstract_lattice
{
    using mlir_dense_abstract_lattice::AbstractDenseLattice;
    pt_map< pt_element > pt_relation;

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

    change_result intersect(const pt_lattice &rhs) {
        change_result res = change_result::NoChange;
        for (const auto &[key, rhs_value] : rhs.pt_relation) {
            auto &lhs_value = pt_relation[key];
            auto to_remove = lhs_value;
            to_remove.set_subtract(rhs_value);
            if (!to_remove.empty()) {
                res |= change_result::Change;
            }
            lhs_value.set_subtract(to_remove);
        }
        return res;
    }

    change_result join(const mlir_dense_abstract_lattice &rhs) override {
        return this->merge(*static_cast< const pt_lattice *>(&rhs));
    };

    mlir::ChangeResult meet(const mlir_dense_abstract_lattice &rhs) override {
        return this->intersect(*static_cast< const pt_lattice *>(&rhs));
    };

    void print(llvm::raw_ostream &os) const override
    {
        for (const auto &[key, vals] : pt_relation) {
            os << key << " -> {";
            std::string sep;
            for (const auto &val : vals) {
                    os << sep << val;
                    sep = ", ";
            }
        }
    }
};

struct pt_analysis : mlir_dense_dfa< pt_lattice >
{
    using mlir_dense_dfa< pt_lattice >::DenseDataFlowAnalysis;

    void visit_pt_op(pt::AddressOfOp &op, const pt_lattice &before, pt_lattice *after) {
        after->join(before);
        auto &lhs_pt = after->pt_relation[op.getLhs()];
        lhs_pt.insert({op.getRhs()});
    };

    void visit_pt_op(pt::AssignOp &op, const pt_lattice &before, pt_lattice *after) {
        after->join(before);

        auto &lhs_pt = after->pt_relation[op.getLhs()];
        const auto &rhs_pt = before.pt_relation.find(op.getRhs())->getSecond();
        for (auto &lhs_val : lhs_pt) {
            auto &insert_point = after->pt_relation[lhs_val];
            insert_point.clear();
            insert_point.set_union(rhs_pt);
        }
    };

    void visit_pt_op(pt::CopyOp &op, const pt_lattice &before, pt_lattice *after) {
        after->join(before);

        auto &lhs_pt = after->pt_relation[op.getLhs()];
        const auto &rhs_pt = before.pt_relation.find(op.getRhs())->getSecond();

        lhs_pt.clear();
        lhs_pt.set_union(rhs_pt);
    };

    void visit_pt_op(pt::DereferenceOp &op, const pt_lattice &before, pt_lattice *after) {
        after->join(before);
        auto &lhs_pt = after->pt_relation[op.getLhs()];
        const auto &rhs_pt = before.pt_relation.find(op.getRhs())->getSecond();
        for (auto &rhs_val : rhs_pt) {
            lhs_pt.set_union(before.pt_relation.find(rhs_val)->getSecond());
        }

    };

    void visit_pt_op(pt::AllocOp &op, const pt_lattice &before, pt_lattice *after) {
        after->join(before);
        //TODO: something more reasonable has to be inserted into the pt set
        //      probably some custom wrapper around mlir value
        static unsigned int count = 0;
        auto set = llvm::SetVector< pt_element >();
        set.insert({value(), "memory_location" + std::to_string(count++)});
        after->pt_relation.insert({{op.getResult()}, set});
    }

    void visitOperation(mlir::Operation *op, const pt_lattice &before, pt_lattice *after) override {
        return llvm::TypeSwitch< mlir::Operation *, void >(op)
            .Case< pt::AddressOfOp,
                   pt::AssignOp,
                   pt::CopyOp,
                   pt::DereferenceOp,
                   pt::AllocOp
            >([&](auto &pt_op) { visit_pt_op(pt_op, before, after); })
            .Default([&](auto &pt_op) { after->join(before); });
    };

    //void visitCallControlFlowTransfer(mlir::CallOpInterface call,
    //                                  mlir::dataflow::CallControlFlowAction action,
    //                                  const pt_lattice &before,
    //                                  pt_lattice *after) override;

    // Default implementation via join should be fine for us (at least for now)
    //void visitRegionBranchControlFlowTransfer(mlir::RegionBranchOpInterface branch,
    //                                          std::optional< unsigned > regionFrom,
    //                                          std::optional< unsigned > regionTo,
    //                                          const pt_lattice &before,
    //                                          pt_lattice *after) override;
    //
    void setToEntryState(pt_lattice *lattice) override
    {
        // TODO: Check if this makes sense?
        propagateIfChanged(lattice, lattice->join(*lattice));
    }
};

void print_analysis_result(mlir::DataFlowSolver &solver, operation *op, llvm::raw_ostream &os);

} // potato::analysis
