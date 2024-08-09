#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/trad/llvm_aa.hpp"

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

    void llvm_andersen::visit_alloc(mllvm::AllocaOp &op, const llaa_lattice &before, llaa_lattice *after) {
        auto changed = after->join(before);
        if (after->new_var(op.getResult()).second)
            changed |= change_result::Change;
        propagateIfChanged(after, changed);
    }

    void llvm_andersen::visit_store(mllvm::StoreOp &op, const llaa_lattice &before, llaa_lattice *after) {
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

    void llvm_andersen::visitOperation(mlir::Operation *op, const llaa_lattice &before, llaa_lattice *after) {
        return llvm::TypeSwitch< mlir::Operation *, void >(op)
            .Case< mllvm::AllocaOp >([&](auto &op) { visit_alloc(op, before, after); })
            .Case< mllvm::StoreOp >([&](auto &op) { visit_store(op, before, after); })
            .Case< mllvm::LoadOp >([&](auto &op) { visit_load(op, before, after); })
            .Case< mllvm::ConstantOp >([&](auto &op) { visit_constant(op, before, after); })
            .Case< mllvm::ICmpOp, mllvm::FCmpOp >([&](auto &op) { visit_cmp(op, before, after); })
            .Default([&](auto &op) { assert(false); propagateIfChanged(after, after->join(before)); });
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
} // namespace potato::trad::analysis
