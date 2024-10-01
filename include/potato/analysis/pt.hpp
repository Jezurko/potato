#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <mlir/IR/Value.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/SetOperations.h>
#include <llvm/ADT/TypeSwitch.h>
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/ops.hpp"
#include "potato/analysis/lattice.hpp"
#include "potato/analysis/utils.hpp"
#include "potato/util/common.hpp"

#include <cassert>
#include <string>

namespace potato::analysis {

struct aa_lattice : mlir_dense_abstract_lattice
{
    using mlir_dense_abstract_lattice::AbstractDenseLattice;
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

    const pointee_set * lookup(const pt_element &val) const {
        auto it = pt_relation.find(val);
        if (it == pt_relation.end())
            return nullptr;
        return &it->second;
    }

    const pointee_set * lookup(const mlir_value &val) const {
        return lookup({ val, "" });
    }

    lattice_set< pt_element > * lookup(const pt_element &val) {
        auto it = pt_relation.find(val);
        if (it == pt_relation.end())
            return nullptr;
        return &it->second;
    }

    lattice_set< pt_element > * lookup(const mlir_value &val) {
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

    change_result init_at_point(ppoint point) {
        auto args = get_args(point);
        change_result changed;
        for (auto &arg : args) {
            changed|= set_var(arg, new_top_set());
        }
        return changed;
    }

    change_result set_all_unknown() {
        auto changed = change_result::NoChange;
        for (auto &[_, pt_set] : pt_relation) {
            changed |= pt_set.set_top();
        }
        return changed;
    }

    change_result merge(const aa_lattice &rhs) {
        change_result res = change_result::NoChange;
        for (const auto &[key, rhs_value] : rhs.pt_relation) {
            auto &lhs_value = pt_relation[key];
            res |= lhs_value.join(rhs_value);
        }
        return res;
    }

    change_result intersect(const aa_lattice &rhs) {
        change_result res = change_result::NoChange;
        for (const auto &[key, rhs_value] : rhs.pt_relation) {
            // non-existent entry would be considered top, so creating a new entry
            // and intersecting it will create the correct value
            auto &lhs_value = pt_relation[key];
            res |= lhs_value.meet(rhs_value);
        }
        return res;
    }

    change_result join(const mlir_dense_abstract_lattice &rhs) override {
        return this->merge(*static_cast< const aa_lattice *>(&rhs));
    };

    change_result meet(const mlir_dense_abstract_lattice &rhs) override {
        return this->intersect(*static_cast< const aa_lattice *>(&rhs));
    };

    void print(llvm::raw_ostream &os) const override;

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

template< typename pt_lattice >
struct pt_analysis : mlir_dense_dfa< pt_lattice >
{
    using base = mlir_dense_dfa< pt_lattice >;
    using base::base;

    using base::propagateIfChanged;

    void visit_pt_op(pt::AddressOp &op, const pt_lattice &before, pt_lattice *after) {
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
        propagateIfChanged(after, changed);
    };

    void visit_pt_op(pt::GlobalVarOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);
        changed |= after->set_var(pt_lattice::new_symbol(op.getName()), pt_lattice::new_top_set());
        propagateIfChanged(after, changed);
    }

    void visit_pt_op(pt::AssignOp &op, const pt_lattice &before, pt_lattice *after) {
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
            return propagateIfChanged(after, changed);
        }

        if (lhs_pt.is_top()) {
            // TODO: do not access the relation by name
            for (auto &[_, pt_set] : after->pt_relation) {
                changed |= pt_set.join(rhs_pt);
            }
            return propagateIfChanged(after, changed);
        }

        for (auto &lhs_val : lhs_pt.get_set_ref()) {
            auto insert_point = after->lookup(lhs_val);
            // unknown insert point ~ top
            if (!insert_point)
                continue;
            changed |= pt_lattice::pointee_union(*insert_point, rhs_pt);
        }
        propagateIfChanged(after, changed);
    };

    void visit_pt_op(pt::CopyOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);

        auto pt_set = pt_lattice::new_pointee_set();

        for (auto operand : op.getOperands()) {
            auto operand_pt = before.lookup(operand);
            if (operand_pt) {
                std::ignore = pt_lattice::pointee_union(pt_set, *operand_pt);
            }
        }

        changed |= after->set_var(op.getResult(), pt_set);
        propagateIfChanged(after, changed);
    };

    void visit_pt_op(pt::DereferenceOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);

        const auto rhs_pt = before.lookup(op.getPtr());
        if (!rhs_pt || rhs_pt->is_top()) {

            changed |= after->set_var(op.getResult(), pt_lattice::new_top_set());
            propagateIfChanged(after, changed);
            return;
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

        propagateIfChanged(after, changed);
    };

    void visit_pt_op(pt::AllocOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);
        if (after->new_var(op.getResult()).second)
            changed |= change_result::Change;
        propagateIfChanged(after, changed);
    }

    void visit_pt_op(pt::ConstantOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);
        changed |= after->set_var(op.getResult(), pt_lattice::new_pointee_set());
        propagateIfChanged(after, changed);

    }

    void visit_pt_op(pt::ValuedConstantOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);
        // TODO: should this really form a self-loop?
        changed |= after->set_var(op.getResult(), op.getResult());
        propagateIfChanged(after, changed);
    }

    void visit_pt_op(pt::UnknownPtrOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);
        changed |= after->set_var(op.getResult(), pt_lattice::new_top_set());
        propagateIfChanged(after, changed);
    }

    void visit_unrealized_cast(mlir::UnrealizedConversionCastOp &op,
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
        propagateIfChanged(after, changed);
    }

    std::vector< mlir::Operation * > get_function_returns(mlir::FunctionOpInterface func) {
        std::vector< mlir::Operation * > returns;
        for (auto &op : func.getFunctionBody().getOps()) {
            if (op.hasTrait< mlir::OpTrait::ReturnLike >())
                returns.push_back(&op);
        }
        return returns;
    }

    std::vector< const pt_lattice * > get_or_create_for(mlir::Operation * dep, const std::vector< mlir::Operation * > &ops) {
        std::vector< const pt_lattice * > states;
        for (const auto &op : ops) {
            states.push_back(this->template getOrCreateFor< pt_lattice >(dep, op));
        }
        return states;
    }

    void visitOperation(mlir::Operation *op, const pt_lattice &before, pt_lattice *after) override {
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
            ([&](auto &pt_op) { visit_pt_op(pt_op, before, after); })
            .template Case< mlir::UnrealizedConversionCastOp >(
                    [&](auto &cast_op) { visit_unrealized_cast(cast_op, before, after); }
            )
            .Default([&](auto &pt_op) { propagateIfChanged(after, after->join(before)); });
    };

    void visitCallControlFlowTransfer(
        mlir::CallOpInterface call, call_cf_action action,
        const pt_lattice &before, pt_lattice *after
    ) override {
        auto changed = after->join(before);
        auto callee  = call.resolveCallable();
        auto func    = mlir::dyn_cast< mlir::FunctionOpInterface >(callee);

        if (action == call_cf_action::EnterCallee) {

            auto &callee_entry = callee->getRegion(0).front();
            auto callee_args   = callee_entry.getArguments();

            for (const auto &[callee_arg, caller_arg] :
                 llvm::zip_equal(callee_args, call.getArgOperands()))
            {
                const auto &caller_pt_set = *before.lookup(caller_arg);
                changed |= after->join_var(callee_arg, caller_pt_set);
            }
            return propagateIfChanged(after, changed);
        }

        if (action == call_cf_action::ExitCallee) {
            auto call_op = call.getOperation();

            if (!func) {
                changed |= after->set_all_unknown();
                for (auto result : call.getOperation()->getResults()) {
                    changed |= after->set_var(result, pt_lattice::new_top_set());
                }
                return propagateIfChanged(after, changed);
            }

            // Join current state to the start of callee to propagate global variables
            // TODO: explore if we can do this more efficient (maybe not join the whole state?)
            if (auto &fn_body = func.getFunctionBody(); !fn_body.empty()) {
                pt_lattice * callee_state = this->template getOrCreate< pt_lattice >(&*fn_body.begin());
                this->addDependency(callee_state, before.getPoint());
                propagateIfChanged(callee_state, callee_state->join(before));
            }

            // Collect states in return statements
            std::vector< mlir::Operation * >  returns       = get_function_returns(func);
            std::vector< const pt_lattice * > return_states = get_or_create_for(call, returns);

            for (const auto &arg : callee->getRegion(0).front().getArguments()) {
                // TODO: Explore call-site sensitivity by having a specific representation for the arguments?
                // If arg at the start points to TOP, then we know nothing
                auto start_state = this->template getOrCreate< pt_lattice >(&*func.getFunctionBody().begin());
                auto arg_pt = start_state->lookup(arg);
                if (!arg_pt || arg_pt->is_top()) {
                    changed |= after->set_all_unknown();
                    break;
                }

                // go through returns and analyze arg pts, join into call operand pt
                for (auto state : return_states) {
                    auto arg_at_ret = state->lookup(arg);
                    changed |= after->join_var(call_op->getOperand(arg.getArgNumber()), *arg_at_ret);
                }
            }

            for (size_t i = 0; i < call_op->getNumResults(); ++i) {
                auto returns_pt = pt_lattice::new_pointee_set();
                for (const auto &[state, ret] : llvm::zip_equal(return_states, returns)) {
                    auto res_pt = state->lookup(ret->getOperand(i));
                    // TODO: Check this: if the result wasn't found, it means that it is unreachable
                    if (res_pt)
                        std::ignore = returns_pt.join(*res_pt);
                }
                changed |= after->join_var(call_op->getResult(i), std::move(returns_pt));
            }
            return propagateIfChanged(after, changed);
        }

        if (action == call_cf_action::ExternalCallee) {
            // TODO:
            // Try to check for "known" functions
            // Try to resolve function pointer calls? (does it happen here?)
            // Make the set of known functions a customization point?
            return propagateIfChanged(after, changed | after->set_all_unknown());
        }

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
        std::ignore = init_state.init_at_point(point);

        this->propagateIfChanged(lattice, lattice->join(init_state));
    }
};

void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);

void print_analysis_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);

} // potato::analysis
