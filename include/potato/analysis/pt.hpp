#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/CallGraph.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/TypeSwitch.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/context.hpp"
#include "potato/analysis/lattice.hpp"
#include "potato/analysis/utils.hpp"
#include "potato/dialect/ops.hpp"
#include "potato/util/common.hpp"

#include <string>

namespace potato::analysis {

struct aa_lattice
{
    using pointee_set = lattice_set< pt_element >;

    pt_map< pt_element, lattice_set > pt_relation;

    static unsigned int variable_count;
    static unsigned int mem_loc_count;
    unsigned int var_count();
    unsigned int alloc_count();

    std::optional< std::string > var_name = {};
    std::optional< std::string > alloc_name = {};
    std::string get_var_name();
    std::string get_alloc_name();

    // TODO: Probably replace most of the following functions with some custom API that doesn't introduce
    //       so many random return values with iterators and stuff

    const pointee_set *lookup(const pt_element &val) const {
        auto it = pt_relation.find(val);
        if (it == pt_relation.end())
            return nullptr;
        return &it->second;
    }

    const pointee_set *lookup(const mlir_value &val) const {
        return lookup({ val, "" });
    }

    pointee_set *lookup(const pt_element &val) {
        auto it = pt_relation.find(val);
        if (it == pt_relation.end())
            return nullptr;
        return &it->second;
    }

    pointee_set *lookup(const mlir_value &val) {
        return lookup({ val, "" });
    }

    auto &operator[](const pt_element &val) {
        return pt_relation[val];
    }

    static auto new_symbol(const llvm::StringRef name) {
        return pt_element(mlir_value(), name.str());
    }

    static auto new_pointee_set() {
        return pointee_set();
    }

    static auto new_top_set() {
        return pointee_set::make_top();
    }

    static auto pointee_union(pointee_set &trg, const pointee_set &src) {
        return trg.join(src);
    }

    auto new_var(mlir_value val) {
        auto set = pointee_set();
        set.insert({mlir_value(), get_alloc_name()});
        return pt_relation.insert({{val, get_var_name()}, set});
    }

    auto new_var(mlir_value var, const pointee_set& pt_set) {
        return pt_relation.insert({{var, get_var_name()}, pt_set});
    }

    auto new_var(mlir_value var, mlir_value pointee) {
        pointee_set set{};
        auto pointee_it = pt_relation.find({ pointee, "" });
        if (pointee_it == pt_relation.end()) {
            assert((mlir::isa< pt::ConstantOp, pt::ValuedConstantOp >(var.getDefiningOp())));
            set.insert({pointee, get_alloc_name()});
        } else {
            set.insert(pointee_it->first);
        }
        return new_var(var, set);
    }

    change_result set_var(mlir_value val, const pointee_set &set) {
        auto [var, inserted] = new_var(val, set);
        if (inserted) {
            return change_result::Change;
        }
        auto &var_pt_set = var->second;
        if (var_pt_set != set) {
            var_pt_set = set;
            return change_result::Change;
        }
        return change_result::NoChange;
    }

    change_result set_var(mlir_value val, mlir_value pointee) {
        auto [var, inserted] = new_var(val, pointee);
        if (inserted)
            return change_result::Change;
        auto &var_pt_set = var->second;
        auto [pointee_var, _] = new_var(pointee, new_pointee_set());
        auto cmp_set = new_pointee_set();
        cmp_set.insert(pointee_var->first);
        if (var_pt_set != cmp_set) {
            var_pt_set = cmp_set;
            return change_result::Change;
        }
        return change_result::NoChange;
    }

    change_result set_var(pt_element elem, const pointee_set &set) {
        auto [var, inserted] = pt_relation.insert({elem, set});
        if (inserted)
            return change_result::Change;
        if (var->second != set) {
            var->second = set;
            return change_result::Change;
        }
        return change_result::NoChange;
    }

    change_result join_var(mlir_value val, pointee_set &&set) {
        auto val_pt  = lookup(val);
        if (!val_pt) {
            return set_var(val, set);
        }
        return val_pt->join(set);
    }

    change_result join_var(mlir_value val, const pointee_set &set) {
        auto val_pt  = lookup(val);
        if (!val_pt) {
            return set_var(val, set);
        }
        return val_pt->join(set);
    }

    change_result set_all_unknown() {
        auto changed = change_result::NoChange;
        for (auto &[_, pt_set] : pt_relation) {
            changed |= pt_set.set_top();
        }
        return changed;
    }

    change_result join(const aa_lattice &rhs) {
        change_result res = change_result::NoChange;
        for (const auto &[key, rhs_value] : rhs.pt_relation) {
            auto &lhs_value = pt_relation[key];
            res |= lhs_value.join(rhs_value);
        }
        return res;
    }

    change_result meet(const aa_lattice &rhs) {
        change_result res = change_result::NoChange;
        for (const auto &[key, rhs_value] : rhs.pt_relation) {
            // non-existent entry would be considered top, so creating a new entry
            // and intersecting it will create the correct value
            auto &lhs_value = pt_relation[key];
            res |= lhs_value.meet(rhs_value);
        }
        return res;
    }

    void print(llvm::raw_ostream &os) const;

    alias_res alias(auto lhs, auto rhs) const {
        const auto lhs_it = find(lhs);
        const auto rhs_it = find(rhs);
        // If we do not know at least one of the arguments we can not deduce any aliasing information
        // TODO: can this happen with correct usage? Should we emit a warning?
        if (lhs_it == pt_relation.end() || rhs_it() == pt_relation.end())
            return alias_res(alias_kind::MayAlias);

        const auto &lhs_pt = *lhs_it;
        const auto &rhs_pt = *rhs_it;

        if (sets_intersect(*lhs_it, *rhs_it)) {
            if (lhs_pt.size() == 1 && rhs_pt.size() == 1) {
                return alias_res(alias_kind::MustAlias);
            }
            return alias_res(alias_kind::MayAlias);
        }

        return alias_res(alias_kind::NoAlias);
    }
};


template< typename pt_lattice, template < typename > typename ctx_wrapper = call_context_wrapper >
struct pt_analysis : mlir_dense_dfa< ctx_wrapper< pt_lattice > >
{
    using ctxed_lattice = ctx_wrapper< pt_lattice >;

    using base = mlir_dense_dfa< ctxed_lattice >;
    using base::base;

    using base::propagateIfChanged;

    pt_analysis(mlir::DataFlowSolver &solver, call_graph *cg) : base(solver), cg(cg) {};

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

        auto &lhs_pt = [&] () -> pt_lattice::pointee_set & {
            auto lhs_pt = after->lookup(op.getLhs());
            if (lhs_pt) {
                return *lhs_pt;
            }
            auto [it, _] = after->new_var(op.getLhs());
            changed |= change_result::Change;
            return it->second;
        }();

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
            auto insert_point = after->lookup(lhs_val);
            // unknown insert point ~ top
            if (!insert_point)
                continue;
            changed |= pt_lattice::pointee_union(*insert_point, rhs_pt);
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
        changed |= after->set_var(op.getResult(), pt_lattice::new_pointee_set());
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
            auto changed_succ = change_result::NoChange;
            auto succ_state = this->template getOrCreate< ctxed_lattice >(successor);
            for (const auto &[ctx, before_with_cr] : before) {
                const auto &[before, changed_before] = before_with_cr;
                auto &[after, changed_after] = succ_state->get_or_propagate_for_context(ctx);
                for (const auto &[pred_op, succ_arg] :
                    llvm::zip_equal(op.getSuccessorOperands(i).getForwardedOperands(), successor->getArguments())
                ) {
                    auto operand_pt = before.lookup(pred_op);
                    changed_after |= after.join_var(succ_arg, *operand_pt);
                }
                changed_succ |= changed_after;
            }
            propagateIfChanged(succ_state, changed_succ);
        }
        propagateIfChanged(after, changed);
    }

    std::vector< const ctxed_lattice * > get_or_create_for(mlir::Operation * dep, const std::vector< mlir::Operation * > &ops) {
        std::vector< const pt_lattice * > states;
        for (const auto &op : ops) {
            states.push_back(this->template getOrCreateFor< ctxed_lattice >(dep, op));
        }
        return states;
    }

    template< typename visitor_t >
    void default_visitor_wrapper(auto op, const ctxed_lattice &before, ctxed_lattice *after, visitor_t visitor) {
        auto changed = change_result::NoChange;
        for (const auto &[ctx, lattice_with_cr] : before) {
            const auto &[before_lattice, before_changed] = lattice_with_cr;
            // new context is automatically considered as changed
            if (before_changed == change_result::NoChange)
                continue;
            if (auto *after_lattice_with_cr = after->get_for_context(ctx)) {
                auto &[after_lattice, after_changed] = *after_lattice_with_cr;
                after_changed |= visitor(op, before_lattice, &after_lattice);
                changed |= after_changed;
            } else {
                auto &[after_lattice, after_changed] = after->propagate_context(ctx, before_lattice);
                after_changed |= visitor(op, before_lattice, &after_lattice);
                changed |= after_changed;
            }
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
        auto caller_node = cg->lookupNode(call->getParentRegion());
        auto edge        = [&]() {
            for (auto &edge : *caller_node) {
                if (&func.getFunctionBody() == edge.getTarget()->getCallableRegion())
                    return edge;
            }
            assert(false);
        }();

        // - `action == CallControlFlowAction::Enter` indicates that:
        //   - `before` is the state before the call operation;
        //   - `after` is the state at the beginning of the callee entry block;
        if (action == call_cf_action::EnterCallee) {
            auto &callee_entry = callee->getRegion(0).front();
            auto callee_args   = callee_entry.getArguments();

            for (const auto &[ctx, lat_with_cr] : before) {
                const auto &before_pt = lat_with_cr.first;
                auto &[after_pt, pt_changed] = after->add_new_context(ctx, {caller_node, edge}, before_pt);
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
                auto &[entry_pt, entry_pt_cr] = callee_entry->add_new_context(
                                                                ctx,
                                                                {caller_node, edge},
                                                                pre_call_pt
                                                              );

                // If both have NoChange it means that the context was already present
                // and that the predecessor in the relevant context didn't change
                // We can safely skip to the next iteration
                if ((entry_pt_cr | pre_call_cr) == change_result::NoChange)
                    continue;

                // Join in to proapgate the state of globals
                entry_pt_cr |= entry_pt.join(pre_call_pt);
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
            propagateIfChanged(callee_entry, entry_changed);

            // Manage the callee exit

            for (auto &[after_ctx, after_with_cr] : *after) {
                auto &[after_pt, after_pt_changed] = after_with_cr;
                auto context = after_ctx;
                context.push_back({caller_node, edge});
                // We won't find the recently added context here
                // But the start of the function was changed, meaning we will propagate
                // to this point again
                if (const auto *pt_ret_state = before.get_for_context(context)) {
                    after_pt_changed |= after_pt.join(pt_ret_state->first);
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

    private:
        call_graph *cg;
};

void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);

void print_analysis_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);

void print_analysis_func_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);
} // potato::analysis
