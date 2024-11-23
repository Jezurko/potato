#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/CallGraph.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/TypeSwitch.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/config.hpp"
#include "potato/analysis/function_models.hpp"
#include "potato/dialect/ops.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis {

template< typename pt_lattice >
struct pt_analysis : mlir_dense_dfa< pt_lattice >
{
    using base = mlir_dense_dfa< pt_lattice >;
    using base::base;

    using base::propagateIfChanged;

    change_result visit_pt_op(pt::AddressOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);
        auto val = op.getVal();

        if (val) {
            changed |= after->join_var(op.getPtr(), op.getVal());
        } else {
            auto symbol_ref = op.getSymbol();
            assert(symbol_ref && "Address of op without value or proper attribute.");

            auto symbol = symbol_table::lookupNearestSymbolFrom(
                op.getOperation(),
                op.getSymbolAttr()
            );

            if (mlir::isa< mlir::FunctionOpInterface >(symbol)) {
                changed |= after->join_var(op.getPtr(), pt_lattice::new_func(symbol));
            }

            if (mlir::isa< pt::GlobalVarOp >(symbol)) {
                changed |= after->join_var(op.getPtr(), pt_lattice::new_glob(symbol));
            }
        }
        return changed;
    };

    change_result visit_pt_op(pt::GlobalVarOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);
        auto &init = op.getInit();
        if (!init.empty()) {
            auto *ret_op = &init.back().back();
            if (ret_op->hasTrait< mlir::OpTrait::ReturnLike >()) {
                auto ret_state = this->template getOrCreate< pt_lattice >(ret_op);
                ret_state->addDependency(after->getPoint(), this);
                propagateIfChanged(ret_state, ret_state->join(before));
                for (auto ret_arg : ret_op->getOperands()) {
                    auto arg_pt = ret_state->lookup(ret_arg);
                    if (arg_pt) {
                        changed |= after->join_var(pt_lattice::new_glob(op.getOperation()), arg_pt);
                    }
                }
                return changed;
            }
        }
        changed |= after->join_var(pt_lattice::new_glob(op.getOperation()), pt_lattice::new_top_set());
        return changed;
    }

    change_result visit_pt_op(pt::AssignOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);

        auto lhs = after->lookup(op.getLhs());
        if (!lhs) {
            return changed;
        }

        const auto rhs = after->lookup(op.getRhs());
        if (!rhs) {
            return changed;
        }

        if (lhs->is_top()) {
            // all pointers may alias, because they contain rhs
            return after->set_all_unknown();
        }

        changed |= after->join_all_pointees_with(lhs, rhs);

        if (changed == change_result::Change) {
            pt_lattice::propagate_members_changed(lhs, get_or_create(), propagate());
        }

        return changed;
    };

    change_result visit_pt_op(pt::CopyOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);

        for (auto operand : op.getOperands()) {
            auto operand_pt = after->lookup(operand);
            if (operand_pt) {
                changed |= after->join_var(op.getResult(), operand_pt);
            }
        }

        return changed;
    };

    change_result visit_pt_op(pt::DereferenceOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);

        const auto rhs_pt = after->lookup(op.getPtr());
        if (!rhs_pt) {
            return changed;
        }
        if (rhs_pt->is_top()) {
            changed |= after->join_var(op.getResult(), after->new_top_set());
            return changed;
        }
        changed |= after->copy_all_pts_into({op.getResult()}, rhs_pt);

        pt_lattice::depend_on_members(rhs_pt, add_dep(after->getPoint()));

        return changed;
    };

    change_result visit_pt_op(pt::AllocOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);
        changed |= after->new_alloca(op.getResult());
        return changed;
    }

    change_result visit_pt_op(pt::ConstantOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);
        changed |= after->add_constant(op.getResult());
        return changed;

    }

    change_result visit_pt_op(pt::UnknownPtrOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);
        changed |= after->join_var(op.getResult(), pt_lattice::new_top_set());
        return changed;
    }

    change_result visit_unrealized_cast(mlir::UnrealizedConversionCastOp &op,
                               const pt_lattice &before, pt_lattice *after)
    {
        auto changed = after->join(before);

        for (auto operand : op.getOperands()) {
            auto operand_pt = after->lookup(operand);
            if (operand_pt) {
                for (auto res : op.getResults()) {
                    changed |= after->join_var(res, operand_pt);
                }
            }
        }
        return changed;
    }

    void visit_branch_interface(mlir::BranchOpInterface &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);

        for (const auto &[i, successor] : llvm::enumerate(op->getSuccessors())) {
            for (const auto &[pred_op, succ_arg] :
                llvm::zip_equal(op.getSuccessorOperands(i).getForwardedOperands(), successor->getArguments())) {
                    auto operand_pt = after->lookup(pred_op);
                    if (!operand_pt) {
                        continue;
                    }
                    changed |= after->join_var(succ_arg, operand_pt);
            }
        }
        propagateIfChanged(after, changed);
    }

    auto get_or_create() {
        return [this](auto arg) -> pt_lattice * {
            return this->template getOrCreate< pt_lattice >(arg);
        };
    }

    auto add_dep(ppoint dep) {
        return [=, this](auto dep_on) {
            auto dep_on_state = this->template getOrCreate< pt_lattice >(dep_on);
            dep_on_state->addDependency(dep, this);
        };
    }

    auto propagate() {
        return [this](pt_lattice *lattice, change_result change) -> void {
            propagateIfChanged(lattice, change);
        };
    }

    void visitOperation(mlir::Operation *op, const pt_lattice &before, pt_lattice *after) override {
        pt_lattice::add_dependencies(op, this, after->getPoint(), get_or_create());

        return llvm::TypeSwitch< mlir::Operation *, void >(op)
            .Case< pt::AddressOp,
                   pt::AllocOp,
                   pt::AssignOp,
                   pt::ConstantOp,
                   pt::CopyOp,
                   pt::DereferenceOp,
                   pt::GlobalVarOp,
                   pt::UnknownPtrOp >
            ([&](auto &pt_op) { auto changed = visit_pt_op(pt_op, before, after); propagateIfChanged(after, changed); })
            .template Case< mlir::UnrealizedConversionCastOp >(
                    [&](auto &cast_op) { auto changed = visit_unrealized_cast(cast_op, before, after); propagateIfChanged(after, changed); }
            )
            .template Case< mlir::BranchOpInterface >([&](auto &branch_op) { visit_branch_interface(branch_op, before, after); })
            .Default([&](auto &pt_op) { propagateIfChanged(after, after->join(before)); });
    };

    change_result visit_function_at_exit(const pt_lattice &before, pt_lattice *after, mlir_operation *callee, mlir::CallOpInterface call) {
        auto changed = change_result::NoChange;
        auto &callee_entry = callee->getRegion(0).front();
        auto callee_args   = callee_entry.getArguments();

        for (const auto &[callee_arg, caller_arg] :
             llvm::zip_equal(callee_args, call.getArgOperands()))
        {
            if (auto arg_pt = after->lookup(caller_arg))
                changed |= after->join_var(callee_arg, arg_pt);
        }
        if constexpr(pt_lattice::propagate_call_arg_zip()) {
            propagateIfChanged(this->template getOrCreate< pt_lattice >(&callee_entry), changed);
        }

        // Manage the callee exit
        if (auto before_exit = mlir::dyn_cast< mlir::Operation * >(before.getPoint());
                 before_exit && before_exit->template hasTrait< mlir::OpTrait::ReturnLike>()
        ) {
            for (size_t i = 0; i < call->getNumResults(); i++) {
                auto res_arg = before_exit->getOperand(i);
                if (auto res_pt = after->lookup(res_arg)) {
                    changed |= after->join_var(call->getResult(i), res_pt);
                }
                if (pt_lattice::propagate_assign()) {
                    pt_lattice *dep_on_state = this->template getOrCreate< pt_lattice >(res_arg.getDefiningOp());
                    dep_on_state->addDependency(after->getPoint(), this);
                }
            }
        }
        return changed;
    }

    change_result visit_function_model(pt_lattice *after, const function_model &model, mlir::CallOpInterface call) {
        auto changed = change_result::NoChange;
        std::vector< mlir_value > copy_from;
        std::vector< mlir_value > copy_to;
        for (size_t i = 0; i < model.args.size(); i++) {
            auto arg_changed = change_result::NoChange;
            switch(model.args[i]) {
                case arg_effect::none:
                    break;
                case arg_effect::alloc:
                    arg_changed |= after->new_alloca(call->getOperand(i));
                    break;
                case arg_effect::copy_src:
                    copy_from.push_back(call->getOperand(i));
                    break;
                case arg_effect::copy_trg:
                    copy_to.push_back(call->getOperand(i));
                    break;
                case arg_effect::unknown:
                    arg_changed |= after->join_var(call->getOperand(i), pt_lattice::new_top_set());
            }
            if constexpr (pt_lattice::propagate_assign()) {
                propagateIfChanged(
                    this->template getOrCreate< pt_lattice >(call->getOperand(i).getDefiningOp()),
                    arg_changed
                );
            }
            changed |= arg_changed;
        }
        // TODO: we only represent single return funtions right now
        // adding multiple-returns should not be complicated
        for (auto res : call->getResults()) {
            switch (model.ret) {
                case ret_effect::none:
                    break;
                case ret_effect::alloc:
                    changed |= after->new_alloca(res);
                    break;
                case ret_effect::copy_trg:
                    copy_to.push_back(res);
                    break;
                case ret_effect::unknown:
                    changed |= after->join_var(res, pt_lattice::new_top_set());
                    break;
            }
        }
        for (const auto &trg : copy_to) {
            for (const auto &src : copy_from) {
                if (auto src_pt = after->lookup(src); src_pt) {
                    auto trg_changed = after->join_var(trg, src_pt);
                    if constexpr (pt_lattice::propagate_assign()) {
                        if (auto def_op = trg.getDefiningOp(); def_op != call.getOperation()) {
                            propagateIfChanged(
                                this->template getOrCreate< pt_lattice >(def_op),
                                trg_changed
                            );
                        }
                    }
                    changed |= trg_changed;
                }
            }
        }
        return changed;
    }

    void visitCallControlFlowTransfer(
        mlir::CallOpInterface call, call_cf_action action,
        const pt_lattice &before, pt_lattice *after
    ) override {
        auto changed     = after->join(before);
        auto callee      = call.resolveCallable();

        // - `action == CallControlFlowAction::Enter` indicates that:
        //   - `before` is the state before the call operation;
        //   - `after` is the state at the beginning of the callee entry block;
        if (action == call_cf_action::EnterCallee) {
            auto &callee_entry = callee->getRegion(0).front();
            auto callee_args   = callee_entry.getArguments();

            for (const auto &[callee_arg, caller_arg] :
                 llvm::zip_equal(callee_args, call.getArgOperands()))
            {
                auto arg_pt = after->lookup(caller_arg);
                changed |= after->join_var(callee_arg, arg_pt);
            }

            return propagateIfChanged(after, changed);
        }

        // - `action == CallControlFlowAction::Exit` indicates that:
        //   - `before` is the state at the end of a callee exit block;
        //   - `after` is the state after the call operation.
        if (action == call_cf_action::ExitCallee) {
            changed |= visit_function_at_exit(before, after, callee, call);
            return propagateIfChanged(after, changed);
        }

        if (action == call_cf_action::ExternalCallee) {
            auto callable = call.getCallableForCallee();
            if (auto symbol = mlir::dyn_cast< mlir::SymbolRefAttr >(callable)) {
                if (auto model_it = models.find(symbol.getLeafReference()); model_it != models.end()) {
                    changed |= visit_function_model(after, model_it->second, call);
                }
            }
            if (auto val = mlir::dyn_cast< mlir_value >(callable)) {
                changed |= after->resolve_fptr_call(
                    val, call, models, get_or_create(), add_dep(after->getPoint()), propagate()
                );
            }
            propagateIfChanged(after, changed );
        }
    };

    // Default implementation via join should be fine for us (at least for now)
    //void visitRegionBranchControlFlowTransfer(mlir::RegionBranchOpInterface branch,
    //                                          std::optional< unsigned > regionFrom,
    //                                          std::optional< unsigned > regionTo,
    //                                          const ctxed_lattice &before,
    //                                          ctxed_lattice *after) override;

    void setToEntryState(pt_lattice *lattice) override {
        if (!lattice->initialized()) {
            lattice->initialize_with(relation);
            propagateIfChanged(lattice, change_result::Change);
        }
    }

    mlir::LogicalResult initialize(mlir_operation *op) override {
        auto state = this->template getOrCreate< pt_lattice >(op);
        state->initialize_with(relation);
        return base::initialize(op);
    }

    void print(llvm::raw_ostream &os) {
        relation->print(os);
    }

    pt_analysis(mlir::DataFlowSolver &solver)
        : base(solver),
          relation(std::make_shared< typename pt_lattice::relation_t >()),
          models(load_and_parse(pointsto_analysis_config))
        {}

    pt_analysis(mlir::DataFlowSolver &solver, std::string config)
        : base(solver),
          relation(std::make_shared< typename pt_lattice::relation_t >()),
          models(load_and_parse(config))
        {}

    private:
    std::shared_ptr< typename pt_lattice::relation_t > relation;
    function_models models;

};

void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);

void print_analysis_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);

void print_analysis_func_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);
} // potato::analysis
