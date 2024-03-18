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

    static unsigned int mem_loc_count;
    unsigned int alloc_count();

    auto new_var(mlir_value val) {
        auto set = llvm::SetVector< pt_element >();
        auto count = alloc_count();
        set.insert({mlir_value(), "mem_loc" + std::to_string(count)});
        return pt_relation.insert({{val, "var" + std::to_string(count)}, set});
    }

    auto new_var(mlir_value var, const llvm::SetVector< pt_element >& pt_set) {
        auto set = llvm::SetVector< pt_element >();
        auto count = alloc_count();
        return pt_relation.insert({{var, "var" + std::to_string(count)}, pt_set});
    }

    void init_at_point(ppoint point) {
        auto args = get_args(point);
        for (auto &arg : args) {
            new_var(arg);
        }
    }

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

    auto find(const mlir_value &val) const {
        return pt_relation.find({val, ""});
    }

    auto find(const pt_element &val) const {
        return pt_relation.find(val);
    }

    auto &operator[](const mlir_value &val) {
        return pt_relation[{val, ""}];
    }

    auto &operator[](const pt_element &val) {
        return pt_relation[val];
    }

    void print(llvm::raw_ostream &os) const override
    {
        for (const auto &[key, vals] : pt_relation) {
            os << key << " -> {";
            std::string sep;
            for (const auto &val : vals) {
                    os << sep << val;
                    sep = ", ";
            }
            os << "}";
        }
    }
};

struct pt_analysis : mlir_dense_dfa< pt_lattice >
{
    using mlir_dense_dfa< pt_lattice >::DenseDataFlowAnalysis;

    void visit_pt_op(pt::AddressOfOp &op, const pt_lattice &before, pt_lattice *after) {
        after->join(before);
        auto &lhs_pt = (*after)[op.getLhs()];
        lhs_pt.clear();
        auto rhs_elem = before.find(op.getRhs())->getFirst();
        lhs_pt.insert(pt_element(rhs_elem));
    };

    void visit_pt_op(pt::AssignOp &op, const pt_lattice &before, pt_lattice *after) {
        after->join(before);

        auto &lhs_pt = (*after)[op.getLhs()];
        const auto &rhs_pt = before.find(op.getRhs())->getSecond();
        for (auto &lhs_val : lhs_pt) {
            auto &insert_point = after->pt_relation[lhs_val];
            insert_point.clear();
            insert_point.set_union(rhs_pt);
        }
    };

    void visit_pt_op(pt::CopyOp &op, const pt_lattice &before, pt_lattice *after) {
        after->join(before);

        auto &lhs_pt = (*after)[op.getLhs()];
        const auto &rhs_pt = before.find(op.getRhs())->getSecond();

        lhs_pt.clear();
        lhs_pt.set_union(rhs_pt);
    };

    void visit_pt_op(pt::DereferenceOp &op, const pt_lattice &before, pt_lattice *after) {
        after->join(before);
        auto &lhs_pt = (*after)[op.getLhs()];
        lhs_pt.clear();
        const auto &rhs_pt = before.find(op.getRhs())->getSecond();
        for (auto &rhs_val : rhs_pt) {
            if (!before.pt_relation.contains(rhs_val)) {
                llvm::errs() << "[PoTATo] Dereferencing a value that points to nothing. Possible bug or error\n";
            }
            lhs_pt.set_union(before.pt_relation.lookup(rhs_val));
        }

    };

    void visit_pt_op(pt::AllocOp &op, const pt_lattice &before, pt_lattice *after) {
        after->join(before);
        after->new_var(op.getResult());
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

    void visitCallControlFlowTransfer(mlir::CallOpInterface call,
                                      mlir::dataflow::CallControlFlowAction action,
                                      const pt_lattice &before,
                                      pt_lattice *after) override {
        if (action == mlir::dataflow::CallControlFlowAction::EnterCallee) {
            auto callee = call.resolveCallable();
            auto &callee_entry = callee->getRegion(0).front();
            auto callee_args = callee_entry.getArguments();

            for (const auto &[callee_arg, caller_arg] : llvm::zip_equal(callee_args, call.getArgOperands())) {
                const auto &pt_set = before.find(caller_arg)->second;
                after->new_var(callee_arg, pt_set);
            }
        }

        if (action == mlir::dataflow::CallControlFlowAction::ExitCallee) {
            for (auto result : call.getOperation()->getResults())
                after->new_var(result);
        }

        after->join(before);
    };

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
        ppoint point = lattice->getPoint();
        auto init_state = pt_lattice(point);
        init_state.init_at_point(point);

        propagateIfChanged(lattice, lattice->join(init_state));
    }
};

void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);

} // potato::analysis
