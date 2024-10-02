#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/trad/llvm_aa.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis::trad {

    unsigned int llaa_lattice::variable_count = 0;
    unsigned int llaa_lattice::mem_loc_count = 0;

    unsigned int llaa_lattice::var_count() { return variable_count++; }

    unsigned int llaa_lattice::alloc_count() { return mem_loc_count++; }

    std::string llaa_lattice::get_var_name() {
        if (!var_name)
            var_name = "var" + std::to_string(var_count());
        return var_name.value();
    }

    std::string llaa_lattice::get_alloc_name() {
        if (!alloc_name)
            alloc_name = "mem_alloc" + std::to_string(alloc_count());
        return alloc_name.value();
    }

    change_result llaa_lattice::join(const mlir::dataflow::AbstractDenseLattice &rhs) {
        change_result res = change_result::NoChange;
        for (const auto &[key, rhs_value] : static_cast< const llaa_lattice *>(&rhs)->pt_relation) {
            auto &lhs_value = pt_relation[key];
            res |= lhs_value.join(rhs_value);
        }
        return res;
    }

    change_result llaa_lattice::meet(const mlir::dataflow::AbstractDenseLattice &rhs) {
        change_result res = change_result::NoChange;
        for (const auto &[key, rhs_value] : static_cast< const llaa_lattice *>(&rhs)->pt_relation) {
            // non-existent entry would be considered top, so creating a new entry
            // and intersecting it will create the correct value
            auto &lhs_value = pt_relation[key];
            res |= lhs_value.meet(rhs_value);
        }
        return res;
    }

    const llaa_lattice::set_t * llaa_lattice::lookup(const pt_element &val) const {
        auto it = pt_relation.find(val);
        if (it == pt_relation.end())
            return nullptr;
        return &it->second;
    }

    const llaa_lattice::set_t * llaa_lattice::lookup(const mlir_value &val) const {
        return lookup({ val, "" });
    }

    llaa_lattice::set_t * llaa_lattice::lookup(const pt_element &val) {
        auto it = pt_relation.find(val);
        if (it == pt_relation.end())
            return nullptr;
        return &it->second;
    }

    llaa_lattice::set_t * llaa_lattice::lookup(const mlir_value &val) {
        return lookup({ val, "" });
    }

    std::pair< llaa_lattice::relation_t::iterator, bool > llaa_lattice::new_var(mlir_value val) {
        auto set = set_t();
        set.insert({mlir_value(), get_alloc_name()});
        return pt_relation.insert({{val, get_var_name()}, set});
    }

    std::pair< llaa_lattice::relation_t::iterator, bool > llaa_lattice::new_var(
            mlir_value val,
            const set_t &pt_set
    ) {
        return pt_relation.insert({{val, get_var_name()}, pt_set});
    }

    std::pair< llaa_lattice::relation_t::iterator, bool > llaa_lattice::new_var(mlir_value var, mlir_value pointee) {
        llaa_lattice::set_t set{};
        auto pointee_it = pt_relation.find({pointee, ""});
        if (pointee_it == pt_relation.end()) {
            set.insert({pointee, get_alloc_name()});
        } else {
            set.insert(pointee_it->first);
        }
        return new_var(var, set);
    }

    change_result llaa_lattice::join_var(mlir_value val, set_t &&set) {
        auto val_pt  = pt_relation.find({val, ""});
        if (val_pt == pt_relation.end()) {
            return set_var(val, set);
        }
        return val_pt->second.join(set);
    }

    change_result llaa_lattice::join_var(mlir_value val, const set_t &set) {
        auto val_pt  = pt_relation.find({val, ""});
        if (val_pt == pt_relation.end()) {
            return set_var(val, set);
        }
        return val_pt->second.join(set);
    }

    change_result llaa_lattice::set_var(mlir_value val, const set_t &pt_set) {
        auto [var, inserted] = new_var(val, pt_set);
        if (inserted) {
            return change_result::Change;
        }
        auto &var_pt_set = var->second;
        if (var_pt_set != pt_set) {
            var_pt_set = {pt_set};
            return change_result::Change;
        }
        return change_result::NoChange;
    }

    change_result llaa_lattice::set_var(mlir_value val, mlir_value pointee) {
        auto [var, inserted] = new_var(val, pointee);
        if (inserted) {
            return change_result::Change;
        } else {
            auto &var_pt_set = var->second;
            auto [var, inserted] = new_var(pointee, llaa_lattice::set_t());
            auto cmp_set = llaa_lattice::set_t();
            cmp_set.insert(var->first);
            if (var_pt_set != cmp_set) {
                var_pt_set = {cmp_set};
                return change_result::Change;
            }
        }
        return change_result::NoChange;
    }

    change_result llaa_lattice::set_var(pt_element elem, const set_t &set) {
        auto [var, inserted] = pt_relation.insert({elem, set});
        if (inserted)
            return change_result::Change;
        if (var->second != set) {
            var->second = set;
            return change_result::Change;
        }
        return change_result::NoChange;
    }

    change_result llaa_lattice::set_all_unknown() {
        auto changed = change_result::NoChange;
        for (auto &[_, pt_set] : pt_relation) {
            changed |= pt_set.set_top();
        }
        return changed;
    }

    void llaa_lattice::print(llvm::raw_ostream &os) const {
        for (const auto &[key, vals] : pt_relation) {
            os << key << " -> " << vals;
        }
    }

    void llvm_andersen::visit_op(mllvm::AllocaOp &op, const llaa_lattice &before, llaa_lattice *after) {
        auto changed = after->join(before);
        if (after->new_var(op.getResult()).second)
            changed |= change_result::Change;
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::StoreOp &op, const llaa_lattice &before, llaa_lattice *after) {
        auto changed = after->join(before);

        auto &addr_pt = after->pt_relation[{op.getAddr(), ""}];
        const auto &val = before.pt_relation.find({op.getValue(), ""});
        const auto &val_pt = val != before.pt_relation.end() ? val->getSecond()
                                                             : llaa_lattice::set_t::make_top();

        if (val_pt.is_bottom()) {
            return propagateIfChanged(after, changed);
        }

        if (addr_pt.is_top()) {
            for (auto &[_, pt_set] : after->pt_relation) {
                changed |= pt_set.join(val_pt);
            }
            return propagateIfChanged(after, changed);
        }

        for (const auto &addr_val : addr_pt.set) {
            auto &insert_point = after->pt_relation[addr_val];
            changed |= insert_point.join(val_pt);
        };

        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::LoadOp &op, const llaa_lattice &before, llaa_lattice *after) {
        auto changed = after->join(before);

        const auto rhs_pt_it = before.pt_relation.find({op.getAddr(), ""});
        if (rhs_pt_it == before.pt_relation.end() || rhs_pt_it->second.is_top()) {
            changed |= after->set_var(op.getResult(), llaa_lattice::set_t::make_top());
            return propagateIfChanged(after, changed);
        }
        const auto rhs_pt = rhs_pt_it->second;
        auto pointees = llaa_lattice::set_t();
        for (const auto &rhs_val : rhs_pt_it->second.get_set_ref()) {
            auto rhs_it = before.pt_relation.find(rhs_val);
            if (rhs_it != before.pt_relation.end()) {
                std::ignore = pointees.join(rhs_it->second);
            } else {
                std::ignore = pointees.join(llaa_lattice::set_t::make_top());
                break;
            }
        }
        changed |= after->set_var(op.getResult(), pointees);
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::ConstantOp &op, const llaa_lattice &before, llaa_lattice *after) {
        auto changed = after->join(before);
        changed |= after->set_var(op.getResult(), llaa_lattice::set_t());
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::GEPOp &op, const llaa_lattice &before, llaa_lattice *after) {
        auto changed = after->join(before);
        if (op->hasAttr(op.getInboundsAttrName())) {
            changed |= after->set_var(op.getResult(), op.getBase());
        } else {
            changed |= after->set_var(op.getResult(), llaa_lattice::set_t::make_top());
        }
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::AddressOfOp &op, const llaa_lattice &before, llaa_lattice *after) {
        auto changed = after->join(before);
        auto set = llaa_lattice::set_t();
        set.insert(pt_element(mlir::Value(), op.getGlobalName().str()));
        changed |= after->set_var(op.getResult(), set);
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::SExtOp &op, const llaa_lattice &before, llaa_lattice *after) {
        auto changed = after->join(before);
        auto set = llaa_lattice::set_t();
        auto arg_pt_it = before.pt_relation.find({op.getArg(), ""});
        std::ignore = set.join(arg_pt_it->second);
        changed |= after->set_var(op.getResult(), set);
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::GlobalOp &op, const llaa_lattice &before, llaa_lattice *after) {
        auto changed = after->join(before);
        if (auto &init = op.getInitializerRegion(); !init.empty()) {
            auto last_op = mlir::cast< mllvm::ReturnOp >(init.back().back());
            auto back_state = getOrCreate< llaa_lattice >(last_op.getOperation());
            if (const auto ret_pt = back_state->lookup(last_op.getArg())) {
                changed |= after->set_var(pt_element(mlir_value(), op.getSymName().str()), *ret_pt);
                return propagateIfChanged(after, changed);
            }
        }
        changed |= after->set_var(
                              pt_element(mlir_value(), op.getSymName().str()),
                              llaa_lattice::set_t::make_top()
                          );
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::MemcpyOp &op, const llaa_lattice &before, llaa_lattice *after) {
        auto changed = after->join(before);
        auto dst_pt = after->lookup(op.getDst());
        auto src_pt = after->lookup(op.getSrc());
        if (!dst_pt || !src_pt) {
            changed |= after->join_var(op.getDst(), llaa_lattice::set_t::make_top());
        } else {
            changed |= after->join_var(op.getDst(), *src_pt);
        }
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::SelectOp &op, const llaa_lattice &before, llaa_lattice *after) {
        auto changed  = after->join(before);
        auto true_pt  = after->lookup(op.getTrueValue());
        auto false_pt = after->lookup(op.getFalseValue());
        if (!true_pt || !false_pt) {
            changed |= after->set_var(op.getRes(), *true_pt);
        } else {
            changed |= after->set_var(op.getRes(), *true_pt);
            changed |= after->join_var(op.getRes(), *false_pt);
        }
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mlir::BranchOpInterface &op, const llaa_lattice &before, llaa_lattice *after) {
        auto changed = after->join(before);

        for (const auto &[i, successor] : llvm::enumerate(op->getSuccessors())) {
            auto changed_succ = change_result::NoChange;
            auto succ_state = this->template getOrCreate< llaa_lattice >(successor);
            for (const auto &[pred_op, succ_arg] :
                llvm::zip_equal(op.getSuccessorOperands(i).getForwardedOperands(), successor->getArguments())
            ) {
                auto operand_pt = after->pt_relation.find({pred_op, ""});
                changed_succ |= after->join_var(succ_arg, operand_pt->second);
            }
            propagateIfChanged(succ_state, changed_succ);
        }

        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_cmp(mlir::Operation *op, const llaa_lattice &before, llaa_lattice *after) {
        auto changed = after->join(before);
        changed |= after->set_var(op->getResult(0), llaa_lattice::set_t());
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_arith(mlir::Operation *op, const llaa_lattice &before, llaa_lattice *after) {
        auto changed = after->join(before);
        for (auto operand : op->getOperands()) {
            auto operand_it = before.pt_relation.find({operand, ""});
            if (operand_it != before.pt_relation.end()) {
                if (!operand_it->second.is_bottom()) {
                    changed |= after->set_var(op->getResult(0), llaa_lattice::set_t::make_top());
                }
            }
        }
        changed |= after->set_var(op->getResult(0), llaa_lattice::set_t());
        propagateIfChanged(after, changed);
    }

    std::vector< mlir::Operation * > llvm_andersen::get_function_returns(mlir::FunctionOpInterface func) {
        std::vector< mlir::Operation * > returns;
        for (auto &op : func.getFunctionBody().getOps()) {
            if (op.hasTrait< mlir::OpTrait::ReturnLike >())
                returns.push_back(&op);
        }
        return returns;
    }

    std::vector< const llaa_lattice * > llvm_andersen::get_or_create_for(mlir::Operation * dep, const std::vector< mlir::Operation * > &ops) {
        std::vector< const llaa_lattice * > states;
        for (const auto &op : ops) {
            states.push_back(this->template getOrCreateFor< llaa_lattice >(dep, op));
        }
        return states;
    }

    void llvm_andersen::visitOperation(mlir::Operation *op, const llaa_lattice &before, llaa_lattice *after) {
        return llvm::TypeSwitch< mlir::Operation *, void >(op)
            .Case< mllvm::AllocaOp,
                   mlir::BranchOpInterface,
                   mllvm::StoreOp,
                   mllvm::LoadOp,
                   mllvm::ConstantOp,
                   mllvm::GEPOp,
                   mllvm::SExtOp,
                   mllvm::AddressOfOp,
                   mllvm::GlobalOp,
                   mllvm::MemcpyOp,
                   mllvm::SelectOp >
            ([&](auto &op) { visit_op(op, before, after); })
            .Case< mllvm::ICmpOp, mllvm::FCmpOp >([&](auto &op) { visit_cmp(op, before, after); })
            .Case< mllvm::FAddOp,
                   mllvm::FDivOp,
                   mllvm::FMulOp,
                   mllvm::FSubOp,
                   mllvm::FMulAddOp,
                   mllvm::FNegOp,
                   mllvm::FAbsOp,
                   mllvm::AddOp,
                   mllvm::UDivOp,
                   mllvm::SDivOp,
                   mllvm::MulOp,
                   mllvm::SubOp,
                   mllvm::ZExtOp >
            ([&](auto &op) { visit_arith(op, before, after); })
            .Case< mllvm::LLVMFuncOp,
                   mllvm::NoAliasScopeDeclOp,
                   mllvm::ReturnOp >
            ([&](auto &) { propagateIfChanged(after, after->join(before)); })
            .Default([&](auto &op) { op->dump(); assert(false); });
    }

    void llvm_andersen::visitCallControlFlowTransfer(mlir::CallOpInterface call,
                                      mlir::dataflow::CallControlFlowAction action,
                                      const llaa_lattice &before,
                                      llaa_lattice *after) {
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
            if (!func) {
                changed |= after->set_all_unknown();
                for (auto result : call->getResults()) {
                    changed |= after->set_var(result, llaa_lattice::set_t::make_top());
                }
                return propagateIfChanged(after, changed);
            }

            // Join current state to the start of callee to propagate global variables
            // TODO: explore if we can do this more efficient (maybe not join the whole state?)
            if (auto &fn_body = func.getFunctionBody(); !fn_body.empty()) {
                llaa_lattice *callee_state = this->template getOrCreate< llaa_lattice >(&*fn_body.begin());
                this->addDependency(callee_state, before.getPoint());
                propagateIfChanged(callee_state, callee_state->join(before));
            }

            // Collect states in return statements
            std::vector< mlir::Operation * >  returns         = get_function_returns(func);
            std::vector< const llaa_lattice * > return_states = get_or_create_for(call, returns);

            for (const auto &arg : callee->getRegion(0).front().getArguments()) {
                // TODO: Explore call-site sensitivity by having a specific representation for the arguments?
                // If arg at the start points to TOP, then we know nothing
                auto start_state = this->template getOrCreate< llaa_lattice >(&*func.getFunctionBody().begin());
                auto arg_pt = start_state->lookup(arg);
                if (!arg_pt || arg_pt->is_top()) {
                    changed |= after->set_all_unknown();
                    break;
                }

                // go through returns and analyze arg pts, join into call operand pt
                for (auto state : return_states) {
                    auto arg_at_ret = state->lookup(arg);
                    changed |= after->join_var(call->getOperand(arg.getArgNumber()), *arg_at_ret);
                }
            }

            for (size_t i = 0; i < call->getNumResults(); ++i) {
                auto returns_pt = llaa_lattice::set_t();
                for (const auto &[state, ret] : llvm::zip_equal(return_states, returns)) {
                    auto res_pt = state->lookup(ret->getOperand(i));
                    // TODO: Check this: if the result wasn't found, it means that it is unreachable
                    if (res_pt)
                        std::ignore = returns_pt.join(*res_pt);
                }
                changed |= after->join_var(call->getResult(i), std::move(returns_pt));
            }
            return propagateIfChanged(after, changed);
        }

        if (action == call_cf_action::ExternalCallee) {
            // TODO:
            // Try to check for "known" functions
            // Try to resolve function pointer calls? (does it happen here?)
            // Make the set of known functions a customization point?
            for (auto result : call->getResults())
                changed |= after->set_var(result, llaa_lattice::set_t::make_top());
            propagateIfChanged(after, changed | after->set_all_unknown());
        }
    }

    void llvm_andersen::setToEntryState(llaa_lattice *lattice) {
        ppoint point = lattice->getPoint();
        auto init_state = llaa_lattice(point);

        if (auto block = mlir::dyn_cast< mlir_block * >(point); block && block->isEntryBlock()) {
            if (auto fn = mlir::dyn_cast< mlir::FunctionOpInterface >(block->getParentOp())) {
                // setup function args
                // we set to top - this method is called at function entries only when not all callers are known
                for (auto &arg : fn.getArguments()) {
                    std::ignore = init_state.set_var(arg, llaa_lattice::set_t::make_top());
                }

                //join in globals
                auto global_scope = fn->getParentRegion();
                for (auto op : global_scope->getOps< mlir::LLVM::GlobalOp >()) {
                    const auto * var_state = this->template getOrCreateFor< llaa_lattice >(point, op.getOperation());
                    std::ignore = init_state.join(*var_state);
                }
            }
        }

        this->propagateIfChanged(lattice, lattice->join(init_state));
    }

    void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os)
    {
        potato::util::print_analysis_result< llaa_lattice >(solver, op, os);
    }

    void print_analysis_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os)
    {
        potato::util::print_analysis_stats< llaa_lattice >(solver, op, os);
    }
} // namespace potato::trad::analysis
