#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/trad/llvm_aa.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis::trad {

    unsigned int llaa_lattice::mem_loc_count = 0;

    unsigned int llaa_lattice::alloc_count() { return mem_loc_count++; }

    llvm::StringRef llaa_lattice::get_alloc_name() {
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
        return lookup(pt_element(val));
    }

    llaa_lattice::set_t * llaa_lattice::lookup(const pt_element &val) {
        auto it = pt_relation.find(val);
        if (it == pt_relation.end())
            return nullptr;
        return &it->second;
    }

    llaa_lattice::set_t * llaa_lattice::lookup(const mlir_value &val) {
        return lookup(pt_element(val));
    }

    std::pair< llaa_lattice::relation_t::iterator, bool > llaa_lattice::new_var(mlir_value val) {
        auto set = set_t();
        set.insert({get_alloc_name()});
        return pt_relation.insert({{val}, set});
    }

    std::pair< llaa_lattice::relation_t::iterator, bool > llaa_lattice::new_var(
            mlir_value val,
            const set_t &pt_set
    ) {
        return pt_relation.insert({{val}, pt_set});
    }

    std::pair< llaa_lattice::relation_t::iterator, bool > llaa_lattice::new_var(mlir_value var, mlir_value pointee) {
        llaa_lattice::set_t set{};
        auto pointee_it = pt_relation.find({pointee});
        if (pointee_it == pt_relation.end()) {
            set.insert({get_alloc_name()});
        } else {
            set.insert(pointee_it->first);
        }
        return new_var(var, set);
    }

    change_result llaa_lattice::join_var(mlir_value val, set_t &&set) {
        auto val_pt  = pt_relation.find({val});
        if (val_pt == pt_relation.end()) {
            return set_var(val, set);
        }
        return val_pt->second.join(set);
    }

    change_result llaa_lattice::join_var(mlir_value val, const set_t &set) {
        auto val_pt  = pt_relation.find({val});
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

    void llvm_andersen::visit_op(mllvm::AllocaOp &op, const aa_lattice &before, aa_lattice *after) {
        auto changed = after->join(before);
        if (after->new_var(op.getResult()).second)
            changed |= change_result::Change;
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::StoreOp &op, const aa_lattice &before, aa_lattice *after) {
        auto changed = after->join(before);

        auto *addr = before.lookup(op.getAddr());
        if (!addr) {
            return propagateIfChanged(after, changed);
        }
        auto &addr_pt = *addr;

        const auto value = before.lookup(op.getValue());
        if (!value || value->is_bottom()) {
            return propagateIfChanged(after, changed);;
        }

        const auto &value_pt = *value;

        if (addr_pt.is_top()) {
            for (auto &[_, pt_set] : *after->pt_relation) {
                changed |= pt_set.join(value_pt);
            }
            return propagateIfChanged(after, changed);
        }

        std::vector< const typename aa_lattice::elem_t * > to_update;
        for (auto &addr_val : addr_pt.get_set_ref()) {
            to_update.push_back(&addr_val);
        }

        for (auto &key : to_update) {
            changed |= after->join_var(*key, value_pt);
        }

        auto addr_state = this->getOrCreate< aa_lattice >(op.getAddr());
        propagateIfChanged(addr_state, changed);

        return propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::LoadOp &op, const aa_lattice &before, aa_lattice *after) {
        auto changed = after->join(before);

        const auto rhs_pt = before.lookup(op.getAddr());
        if (!rhs_pt || rhs_pt->is_bottom()) {
            // we don't know yet - bail out
            return propagateIfChanged(after, changed);
        }
        if (rhs_pt->is_top()) {
            changed |= after->set_var(op.getResult(), llaa_lattice::set_t::make_top());
            return propagateIfChanged(after, changed);
        }
        std::vector< decltype(before.lookup(typename aa_lattice::elem_t())) > to_join;
        for (const auto &rhs_val : rhs_pt->get_set_ref()) {
            auto rhs_pt = before.lookup(rhs_val);
            if (rhs_pt) {
                to_join.push_back(rhs_pt);
            } else {
            }
        }

        if (to_join.empty()) {
            changed |= after->join_empty(op.getResult());
        }

        for (auto *join : to_join) {
            changed |= after->join_var(op.getResult(), *join);
        }

        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::ConstantOp &op, const aa_lattice &before, aa_lattice *after) {
        auto changed = after->join(before);
        changed |= after->add_constant(op.getResult());
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::ZeroOp &op, const aa_lattice &before, aa_lattice *after) {
        auto changed = after->join(before);
        changed |= after->add_constant(op.getResult());
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::GEPOp &op, const aa_lattice &before, aa_lattice *after) {
        auto changed = after->join(before);
        if (op->hasAttr(op.getInboundsAttrName())) {
            if (auto base_pt = before.lookup(op.getBase()))
                changed |= after->join_var(op.getResult(), *base_pt);
        } else {
            changed |= after->join_var(op.getResult(), aa_lattice::new_top_set());
        }
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::AddressOfOp &op, const aa_lattice &before, aa_lattice *after) {
        auto changed = after->join(before);
        changed |= after->join_var(op.getResult(), aa_lattice::new_symbol(op.getGlobalName()));
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::SExtOp &op, const aa_lattice &before, aa_lattice *after) {
        auto changed = after->join(before);
        if (auto arg_pt = before.lookup(op.getArg())) {
            changed |= after->join_var(op.getResult(), *arg_pt);
        }
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::GlobalOp &op, const aa_lattice &before, aa_lattice *after) {
        auto changed = after->join(before);
        auto &init = op.getInitializer();
        if (!init.empty()) {
            auto *ret_op = &init.back().back();
            if (ret_op->hasTrait< mlir::OpTrait::ReturnLike >()) {
                auto ret_state = this->template getOrCreate< aa_lattice >(ret_op);
                ret_state->addDependency(after->getPoint(), this);
                propagateIfChanged(ret_state, ret_state->join(before));
                for (auto ret_arg : ret_op->getOperands()) {
                    auto *arg_pt = ret_state->lookup(ret_arg);
                    if (arg_pt) {
                        changed |= after->join_var(aa_lattice::new_symbol(op.getName()), *arg_pt);
                    }
                }
                return propagateIfChanged(after, changed);
            }
        }
        changed |= after->join_var(aa_lattice::new_symbol(op.getName()), aa_lattice::new_top_set());
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::MemcpyOp &op, const aa_lattice &before, aa_lattice *after) {
        auto changed = after->join(before);
        if (auto src_pt = after->lookup(op.getSrc())) {
            changed |= after->join_var(op.getDst(), *src_pt);
        }
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mllvm::SelectOp &op, const aa_lattice &before, aa_lattice *after) {
        auto changed = after->join(before);
        if (auto true_pt = before.lookup(op.getTrueValue())) {
            changed |= after->join_var(op.getRes(), *true_pt);
        }
        if (auto false_pt = before.lookup(op.getTrueValue())) {
            changed |= after->join_var(op.getRes(), *false_pt);
        }
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_op(mlir::BranchOpInterface &op, const aa_lattice &before, aa_lattice *after) {
        auto changed = after->join(before);

        for (const auto &[i, successor] : llvm::enumerate(op->getSuccessors())) {
            for (const auto &[pred_op, succ_arg] :
                llvm::zip_equal(op.getSuccessorOperands(i).getForwardedOperands(), successor->getArguments())) {
                    auto operand_pt = after->lookup(pred_op);
                    if (!operand_pt) {
                        continue;
                    }
                    changed |= after->join_var(succ_arg, *operand_pt);
            }
        }
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_cmp(mlir::Operation *op, const aa_lattice &before, aa_lattice *after) {
        auto changed = after->join(before);
        changed |= after->add_constant(op->getResult(0));
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_arith(mlir::Operation *op, const aa_lattice &before, aa_lattice *after) {
        auto changed = after->join(before);
        for (auto operand : op->getOperands()) {
            if (auto operand_it = before.lookup(operand)) {
                changed |= after->join_var(op->getResult(0), *operand_it);
            }
        }
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

    void llvm_andersen::visitOperation(mlir::Operation *op, const aa_lattice &before, aa_lattice *after) {
        return llvm::TypeSwitch< mlir::Operation *, void >(op)
            .Case< mllvm::AllocaOp,
                   mlir::BranchOpInterface,
                   mllvm::StoreOp,
                   mllvm::LoadOp,
                   mllvm::ConstantOp,
                   mllvm::ZeroOp,
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
                   mllvm::FPToSIOp,
                   mllvm::FPToUIOp,
                   mllvm::UIToFPOp,
                   mllvm::SIToFPOp,
                   mllvm::AddOp,
                   mllvm::UDivOp,
                   mllvm::SDivOp,
                   mllvm::MulOp,
                   mllvm::SubOp,
                   mllvm::ZExtOp >
            ([&](auto &op) { visit_arith(op, before, after); })
            .Case< mllvm::LLVMFuncOp,
                   mllvm::NoAliasScopeDeclOp,
                   mllvm::AssumeOp,
                   mllvm::ReturnOp >
            ([&](auto &) { propagateIfChanged(after, after->join(before)); })
            .Default([&](auto &op) { op->dump(); assert(false); });
    }

    void llvm_andersen::visitCallControlFlowTransfer(mlir::CallOpInterface call,
                                      mlir::dataflow::CallControlFlowAction action,
                                      const aa_lattice &before,
                                      aa_lattice *after) {
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
                changed |= after->join_var(callee_arg, *arg_pt);
            }

            return propagateIfChanged(after, changed);
        }

        // - `action == CallControlFlowAction::Exit` indicates that:
        //   - `before` is the state at the end of a callee exit block;
        //   - `after` is the state after the call operation.
        if (action == call_cf_action::ExitCallee) {
            auto &callee_entry = callee->getRegion(0).front();
            auto callee_args   = callee_entry.getArguments();

            for (const auto &[callee_arg, caller_arg] :
                 llvm::zip_equal(callee_args, call.getArgOperands()))
            {
                if (auto arg_pt = after->lookup(caller_arg))
                    changed |= after->join_var(callee_arg, *arg_pt);
            }
            propagateIfChanged(this->template getOrCreate< aa_lattice >(&callee_entry), changed);

            // Manage the callee exit
            if (auto before_exit = mlir::dyn_cast< mlir::Operation * >(before.getPoint());
                     before_exit && before_exit->template hasTrait< mlir::OpTrait::ReturnLike>()
            ) {
                for (size_t i = 0; i < call->getNumResults(); i++) {
                    auto res_arg = before_exit->getOperand(i);
                    if (auto res_pt = after->lookup(res_arg)) {
                        changed |= after->join_var(call->getResult(i), *res_pt);
                    }
                }
            }

            return propagateIfChanged(after, changed);
        }

        if (action == call_cf_action::ExternalCallee) {
            // TODO:
            // Try to check for "known" functions
            // Try to resolve function pointer calls? (does it happen here?)
            // Make the set of known functions a customization point?
            for (auto operand : call.getArgOperands()) {
                //TODO: propagate TOP
            }
            for (auto result : call->getResults()) {
                changed |= after->join_var(result, aa_lattice::new_top_set());
            }
            propagateIfChanged(after, changed );
        }
    }

    void llvm_andersen::setToEntryState(aa_lattice *lattice) {
        if (lattice->pt_relation != relation) {
            lattice->pt_relation = relation;
            propagateIfChanged(lattice, change_result::Change);
        }
    }

    mlir::LogicalResult llvm_andersen::initialize(mlir_operation *op) {
        if (!relation) {
            relation = std::make_shared< typename aa_lattice::relation_t >();
        }
        auto state = this->template getOrCreate< aa_lattice >(op);
        state->pt_relation = relation;
        return base::initialize(op);
    }

    void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os)
    {
        potato::util::print_analysis_result< llaa_lattice >(solver, op, os);
    }
} // namespace potato::trad::analysis
