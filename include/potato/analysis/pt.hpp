#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/CallGraph.h>
#include <mlir/Analysis/DataFlow/DeadCodeAnalysis.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/TypeSwitch.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/context.hpp"
#include "potato/dialect/ops.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis {

template< typename pt_lattice, template < typename, unsigned > typename ctx_wrapper = call_context_wrapper, unsigned ctx_size = 1 >
struct pt_analysis : mlir_dense_dfa< ctx_wrapper< pt_lattice, ctx_size > >
{
    using ctxed_lattice = ctx_wrapper< pt_lattice, ctx_size >;

    using base = mlir_dense_dfa< ctxed_lattice >;
    using base::base;

    using base::propagateIfChanged;

    static change_result visit_pt_op(pt::AddressOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);
        auto val = op.getVal();

        if (val) {
            changed |= after->set_var(op.getPtr(), op.getVal());
        } else {
            auto symbol_ref = op.getSymbol();
            assert(symbol_ref && "Address of op without value or proper attribute.");

            auto pt_set = pt_lattice::new_pointee_set();
            pt_set.insert(pt_lattice::new_symbol(symbol_ref.value()));

            changed |= after->set_var(op.getPtr(), pt_set);
        }
        return changed;
    };

    static change_result visit_pt_op(pt::GlobalVarOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);
        changed |= after->set_var(pt_lattice::new_symbol(op.getName()), pt_lattice::new_top_set());
        return changed;
    }

    static change_result visit_pt_op(pt::AssignOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);

        auto lhs = before.lookup(op.getLhs());
        if (!lhs) {
            return changed;
        }
        const auto &lhs_pt = *lhs;

        const auto rhs = before.lookup(op.getRhs());
        const auto &rhs_pt = rhs ? *rhs : pt_lattice::new_top_set();

        if (rhs_pt.is_bottom()) {
            return changed;
        }

        if (lhs_pt.is_top()) {
            // TODO: do not access the relation by name
            for (auto &[_, pt_set] : after->pt_relation) {
                changed |= pt_set.join(rhs_pt);
            }
            return changed;
        }

        for (auto &lhs_val : lhs_pt.get_set_ref()) {
            changed |= after->join_var(lhs_val, rhs_pt);
        }
        return changed;
    };

    static change_result visit_pt_op(pt::CopyOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);

        auto pt_set = pt_lattice::new_pointee_set();

        for (auto operand : op.getOperands()) {
            auto operand_pt = before.lookup(operand);
            if (operand_pt) {
                std::ignore = pt_lattice::pointee_union(pt_set, *operand_pt);
            }
        }

        changed |= after->set_var(op.getResult(), pt_set);
        return changed;
    };

    static change_result visit_pt_op(pt::DereferenceOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);

        const auto rhs_pt = before.lookup(op.getPtr());
        if (!rhs_pt || rhs_pt->is_top()) {

            changed |= after->set_var(op.getResult(), pt_lattice::new_top_set());
            return changed;
        }

        auto pointees = pt_lattice::new_pointee_set();
        for (const auto &val : rhs_pt->get_set_ref()) {
            auto val_pt = before.lookup(val);
            // We can ignore the change results as they will be resolved by set_var
            if (val_pt) {
                std::ignore = pt_lattice::pointee_union(pointees, *val_pt);
            } else {
                // If we didn't find the value, we should safely assume it can point anywhere
                std::ignore = pt_lattice::pointee_union(pointees, pt_lattice::new_top_set());
                // Further joins won't change anything because we are already top
                break;
            }
        }
        changed |= after->set_var(op.getResult(), pointees);

        return changed;
    };

    static change_result visit_pt_op(pt::AllocOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);
        if (after->new_var(op.getResult()).second)
            changed |= change_result::Change;
        return changed;
    }

    static change_result visit_pt_op(pt::ConstantOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);
        changed |= after->set_var(op.getResult(), pt_lattice::new_top_set());
        return changed;

    }

    static change_result visit_pt_op(pt::ValuedConstantOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);
        // TODO: should this really form a self-loop?
        changed |= after->set_var(op.getResult(), op.getResult());
        return changed;
    }

    static change_result visit_pt_op(pt::UnknownPtrOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);
        changed |= after->set_var(op.getResult(), pt_lattice::new_top_set());
        return changed;
    }

    static change_result visit_unrealized_cast(mlir::UnrealizedConversionCastOp &op,
                               const pt_lattice &before, pt_lattice *after)
    {
        auto changed = after->join(before);

        auto pt_set = pt_lattice::new_pointee_set();

        for (auto operand : op.getOperands()) {
            auto operand_pt = before.lookup(operand);
            // We can ignore the change results as they will be resolved by set_var
            if (operand_pt) {
                std::ignore = pt_lattice::pointee_union(pt_set, *operand_pt);
            } else {
                // If we didn't find the value, we should safely assume it can point anywhere
                std::ignore = pt_lattice::pointee_union(pt_set, pt_lattice::new_top_set());
                // Further joins won't change anything because we are already top
                break;
            }
        }
        for (auto res : op.getResults()) {
            changed |= after->set_var(res, pt_set);
        }
        return changed;
    }

    void visit_branch_interface(mlir::BranchOpInterface &op, const ctxed_lattice &before, ctxed_lattice *after) {
        auto changed = after->join(before);

        for (const auto &[i, successor] : llvm::enumerate(op->getSuccessors())) {
            auto succ_state = this->template getOrCreate< ctxed_lattice >(this->template getProgramPoint< mlir::dataflow::CFGEdge >(op->getBlock(), successor));
            auto changed_succ = succ_state->join(before);
            for (const auto &[ctx, before_with_cr] : before) {
                const auto &[before_pt, changed_before] = before_with_cr;

                auto [succ_with_cr, inserted] = succ_state->add_context(ctx, before_pt);
                auto &[succ, changed_succ_pt] = *succ_with_cr;
                if (!inserted) {
                    changed_succ_pt = succ.join(before_pt);
                }

                for (const auto &[pred_op, succ_arg] :
                    llvm::zip_equal(op.getSuccessorOperands(i).getForwardedOperands(), successor->getArguments())
                ) {
                    auto operand_pt = before_pt.lookup(pred_op);
                    if (!operand_pt)
                        continue;
                    changed_succ_pt |= succ.join_var(succ_arg, *operand_pt);
                }
                changed_succ |= changed_succ_pt;
            }
            propagateIfChanged(succ_state, changed_succ);
        }
        propagateIfChanged(after, changed);
    }

    template< typename visitor_t >
    void default_visitor_wrapper(auto op, const ctxed_lattice &before, ctxed_lattice *after, visitor_t visitor) {
        auto changed = change_result::NoChange;
        for (const auto &[ctx, lattice_with_cr] : before) {
            const auto &[before_lattice, before_changed] = lattice_with_cr;
            // new context is automatically considered as changed
            if (before_changed == change_result::NoChange)
                continue;
            auto [after_with_cr, inserted] = after->add_context(ctx, before_lattice);
            auto &[after_lattice, after_changed] = *after_with_cr;
            if (!inserted)
                after_changed = change_result::NoChange;

            after_changed |= visitor(op, before_lattice, &after_lattice);
            changed |= after_changed;
        }
        propagateIfChanged(after, changed);
    }

    void visitOperation(mlir::Operation *op, const ctxed_lattice &before, ctxed_lattice *after) override {
        return llvm::TypeSwitch< mlir::Operation *, void >(op)
            .Case< pt::AddressOp,
                   pt::AllocOp,
                   pt::AssignOp,
                   pt::ConstantOp,
                   pt::CopyOp,
                   pt::DereferenceOp,
                   pt::GlobalVarOp,
                   pt::ValuedConstantOp,
                   pt::UnknownPtrOp >
            ([&](auto &pt_op) { default_visitor_wrapper< change_result(decltype(pt_op), const pt_lattice &, pt_lattice *) >(pt_op, before, after, visit_pt_op); })
            .template Case< mlir::UnrealizedConversionCastOp >(
                    [&](auto &cast_op) { default_visitor_wrapper(cast_op, before, after, visit_unrealized_cast); }
            )
            .template Case< mlir::BranchOpInterface >([&](auto &branch_op) { visit_branch_interface(branch_op, before, after); })
            .Default([&](auto &pt_op) { propagateIfChanged(after, after->join(before)); });
    };

    void visitCallControlFlowTransfer(
        mlir::CallOpInterface call, call_cf_action action,
        const ctxed_lattice &before, ctxed_lattice *after
    ) override {

        auto changed     = after->join(before);
        auto callee      = call.resolveCallable();
        auto func        = mlir::dyn_cast< mlir::FunctionOpInterface >(callee);

        // - `action == CallControlFlowAction::Enter` indicates that:
        //   - `before` is the state before the call operation;
        //   - `after` is the state at the beginning of the callee entry block;
        if (action == call_cf_action::EnterCallee) {
            auto &callee_entry = callee->getRegion(0).front();
            auto callee_args   = callee_entry.getArguments();

            for (const auto &[ctx, lat_with_cr] : before) {
                const auto &before_pt = lat_with_cr.first;
                auto new_ctx = ctx;
                new_ctx.push_back(call.getOperation());

                auto [after_with_cr, inserted] = after->add_context(std::move(new_ctx), before_pt);
                auto &[after_pt, pt_changed] = *after_with_cr;
                if (!inserted)
                    pt_changed = change_result::NoChange;

                for (const auto &[callee_arg, caller_arg] :
                     llvm::zip_equal(callee_args, call.getArgOperands()))
                {
                    const auto &caller_pt_set = *before_pt.lookup(caller_arg);
                    pt_changed |= after_pt.join_var(callee_arg, caller_pt_set);
                }
                changed |= pt_changed;
            }

            return propagateIfChanged(after, changed);
        }

        // - `action == CallControlFlowAction::Exit` indicates that:
        //   - `before` is the state at the end of a callee exit block;
        //   - `after` is the state after the call operation.
        if (action == call_cf_action::ExitCallee) {
            // Insert with new context into the start of the callee.
            // We can't rely on this being done udner`EnterCallee`, as that version is called only
            // if the analysis framework knows all callsites which is quite rare
            // as most functions can be called outside of the current module
            // We can afford to do this, as we explicitely manage call contexts
            auto state_pre_call = [&](){
                if (mlir_operation *pre_call = call->getPrevNode())
                    return this->template getOrCreate< ctxed_lattice >(pre_call);
                else
                    return this->template getOrCreate< ctxed_lattice >(call->getBlock());
            }();
            // Add dependecy and fetch the state at the start of the callee
            auto callee_entry = this->template getOrCreate< ctxed_lattice >(&*func.begin());
            callee_entry->addDependency(state_pre_call->getPoint(), this);
            auto entry_changed = change_result::NoChange;

            // Add call contexts
            for (const auto &[ctx, lat_with_cr] : *state_pre_call) {
                const auto &[pre_call_pt, pre_call_cr] = lat_with_cr;
                auto new_ctx = ctx;
                new_ctx.push_back(call.getOperation());
                auto [entry_pt_with_cr, inserted] = callee_entry->add_context(
                                                                std::move(new_ctx),
                                                                pre_call_pt
                                                              );
                auto &[entry_pt, entry_pt_cr] = *entry_pt_with_cr;

                if (!inserted) {
                    entry_pt_cr = entry_pt.join(pre_call_pt);
                    // if the ctx already exists, we need to move forward
                    changed |= after->join(*state_pre_call);
                }

                // Add args pt
                auto zipped_args = llvm::zip_equal(func.getArguments(), call.getArgOperands());
                for (const auto &[callee_arg, caller_arg] : zipped_args) {
                    if (auto arg_pt = pre_call_pt.lookup(caller_arg))
                        entry_pt_cr |= entry_pt.set_var(callee_arg, *arg_pt);
                    else
                        entry_pt_cr |= entry_pt.set_var(callee_arg, pt_lattice::new_top_set());
                }
                entry_changed |= entry_pt_cr;
            }
            if (entry_changed == change_result::Change)
                propagateIfChanged(callee_entry, entry_changed);

            // Manage the callee exit

            for (auto &[after_ctx, after_with_cr] : *after) {
                auto &[after_pt, after_pt_changed] = after_with_cr;
                auto context = after_ctx;
                context.push_back(call);
                // We won't find the recently added context here
                // But the start of the function was changed, meaning we will propagate
                // to this point again
                llvm::errs() << "getting for ctx\n";
                if (const auto *pt_ret_state = before.get_for_context(context)) {
                    llvm::errs() << "got for ctx\n";
                    after_pt_changed = after_pt.join(pt_ret_state->first);
                    // hookup results of return
                    if (auto before_exit = mlir::dyn_cast< mlir::Operation * >(before.getPoint());
                             before_exit && before_exit->template hasTrait< mlir::OpTrait::ReturnLike>()
                    ) {
                        for (size_t i = 0; i < call->getNumResults(); i++) {
                            auto res_arg = before_exit->getOperand(i);
                            if (auto res_pt = pt_ret_state->first.lookup(res_arg)) {
                                after_pt_changed |= after_pt.join_var(call->getResult(i), *res_pt);
                            } else {
                                after_pt_changed |= after_pt.join_var(call->getResult(i), pt_lattice::new_top_set());
                            }
                        }
                    }
                }
                changed |= after_pt_changed;
            }
            propagateIfChanged(after, changed);
        }

        if (action == call_cf_action::ExternalCallee) {
            // TODO:
            // Try to check for "known" functions
            // Try to resolve function pointer calls? (does it happen here?)
            // Make the set of known functions a customization point?
            for (auto &[ctx, after_with_cr] : *after) {
                auto &[after_pt, pt_cr] = after_with_cr;
                for (auto result : call->getResults()) {
                    pt_cr |= after_pt.set_var(result, pt_lattice::new_top_set());
                }
                pt_cr |= after_pt.set_all_unknown();
                changed |= pt_cr;
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

    void setToEntryState(ctxed_lattice *lattice) override {
        ppoint point = lattice->getPoint();
        auto &[state, state_changed] = lattice->get_for_default_context();
        if (auto block = mlir::dyn_cast< mlir_block * >(point); block && block->isEntryBlock()) {
            if (auto fn = mlir::dyn_cast< mlir::FunctionOpInterface >(block->getParentOp())) {
                // setup function args
                // we set to top - this method is called at function entries only when not all callers are known
                for (auto &arg : fn.getArguments()) {
                    state_changed |= state.set_var(arg, pt_lattice::new_top_set());
                }

                // join in globals
                // This assumes all functions are defined in the top-level scope
                // It might not be true for all possible users?
                auto global_scope = fn->getParentRegion();
                for (auto op : global_scope->getOps< pt::GlobalVarOp >()) {
                    const auto *var_state = this->template getOrCreateFor< ctxed_lattice >(point, op.getOperation());
                    const auto &[glob_state, _] = var_state->get_for_default_context();
                    state_changed |= state.join(glob_state);
                }
            }
        }
        this->propagateIfChanged(lattice, state_changed);
    }
};

void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);

void print_analysis_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);

void print_analysis_func_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);
} // potato::analysis
