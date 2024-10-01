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

    void llvm_andersen::visitOperation(mlir::Operation *op, const llaa_lattice &before, llaa_lattice *after) {
        return llvm::TypeSwitch< mlir::Operation *, void >(op)
            .Case< mllvm::AllocaOp,
                   mllvm::StoreOp,
                   mllvm::LoadOp,
                   mllvm::ConstantOp,
                   mllvm::GEPOp,
                   mllvm::SExtOp,
                   mllvm::AddressOfOp >
            ([&](auto &op) { visit_op(op, before, after); })
            .Case< mllvm::ICmpOp, mllvm::FCmpOp >([&](auto &op) { visit_cmp(op, before, after); })
            .Case< mllvm::FAddOp,
                   mllvm::FDivOp,
                   mllvm::FMulOp,
                   mllvm::FSubOp,
                   mllvm::FMulAddOp,
                   mllvm::FNegOp,
                   mllvm::AddOp,
                   mllvm::UDivOp,
                   mllvm::SDivOp,
                   mllvm::MulOp,
                   mllvm::SubOp >
            ([&](auto &op) { visit_arith(op, before, after); })
            .Case< mllvm::GlobalOp,
                   mllvm::LLVMFuncOp,
                   mllvm::BrOp,
                   mllvm::CondBrOp,
                   mllvm::ReturnOp >
            ([&](auto &) { propagateIfChanged(after, after->join(before)); })
            .Default([&](auto &op) { op->dump(); assert(false); });
    }

    void llvm_andersen::visitCallControlFlowTransfer(mlir::CallOpInterface call,
                                      mlir::dataflow::CallControlFlowAction action,
                                      const llaa_lattice &before,
                                      llaa_lattice *after) {
        if (action == mlir::dataflow::CallControlFlowAction::EnterCallee) {
            auto callee = call.resolveCallable();
            auto &callee_entry = callee->getRegion(0).front();
            auto callee_args = callee_entry.getArguments();

            for (const auto &[callee_arg, caller_arg] : llvm::zip_equal(callee_args, call.getArgOperands())) {
                const auto &pt_set = before.pt_relation.find({caller_arg, ""})->second;
                after->new_var(callee_arg, pt_set);
            }
        }

        if (action == mlir::dataflow::CallControlFlowAction::ExitCallee) {
            for (auto result : call.getOperation()->getResults())
                after->new_var(result);
        }

        propagateIfChanged(after, after->join(before));
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
