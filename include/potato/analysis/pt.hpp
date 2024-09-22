#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Interfaces/CallInterfaces.h>
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

    auto find(const mlir_value &val) const {
        return pt_relation.find({val, ""});
    }

    auto find(const pt_element &val) const {
        return pt_relation.find(val);
    }

    auto find(const mlir_value &val) {
        return pt_relation.find({val, ""});
    }

    auto find(const pt_element &val) {
        return pt_relation.find(val);
    }

    const lattice_set< pt_element > * lookup(const pt_element &val) const {
        auto it = pt_relation.find(val);
        if (it == pt_relation.end())
            return nullptr;
        return &it->second;
    }

    const lattice_set< pt_element > * lookup(const mlir_value &val) const {
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
        auto pointee_it = find(pointee);
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
            var_pt_set = {set};
            return change_result::Change;
        }
        return change_result::NoChange;
    }

    change_result set_var(mlir_value val, mlir_value pointee) {
        auto [var, inserted] = new_var(val, pointee);
        if (inserted) {
            return change_result::Change;
        } else {
            auto &var_pt_set = var->second;
            auto [var, inserted] = new_var(pointee, new_pointee_set());
            auto cmp_set = new_pointee_set();
            cmp_set.insert(var->first);
            if (var_pt_set != cmp_set) {
                var_pt_set = {cmp_set};
                return change_result::Change;
            }
        }
        return change_result::NoChange;
    }

    change_result join_var(mlir_value val, pointee_set &&set) {
        auto val_it = pt_relation.find({val, ""});
        if (val_it == pt_relation.end()) {
            return set_var(val, set);
        }
        return val_it->second.join(set);
    }

    change_result join_var(mlir_value val, const pointee_set &set) {
        auto val_it = pt_relation.find({val, ""});
        if (val_it == pt_relation.end()) {
            return set_var(val, set);
        }
        return val_it->second.join(set);
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

    auto end() const { return pt_relation.end(); }

    void print(llvm::raw_ostream &os) const override;

    alias_res alias(auto lhs, auto rhs) const {
        const auto lhs_it = find(lhs);
        const auto rhs_it = find(rhs);
        // If we do not know at least one of the arguments we can not deduce any aliasing information
        // TODO: can this happen with correct usage? Should we emit a warning?
        if (lhs_it == end() || rhs_it() == end())
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
            auto symbol_ref = op->getAttrOfType< mlir::FlatSymbolRefAttr >("addr_of");
            assert(symbol_ref && "Address of op without value or proper attribute.");

            auto pt_set = pt_lattice::new_pointee_set();
            pt_set.insert(pt_lattice::new_symbol(symbol_ref.getValue()));

            changed |= after->set_var(op.getPtr(), pt_set);
        }
        propagateIfChanged(after, changed);
    };

    void visit_pt_op(pt::AssignOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);

        auto &lhs_pt = (*after)[op.getLhs()];
        const auto &rhs = before.find(op.getRhs());
        const auto &rhs_pt = rhs != before.end() ? rhs->getSecond() : pt_lattice::new_top_set();

        if (rhs_pt.is_bottom()) {
            return propagateIfChanged(after, changed);
        }

        if (lhs_pt.is_top()) {
            for (auto &[_, pt_set] : after->pt_relation) {
                changed |= pt_set.join(rhs_pt);
            }
            return propagateIfChanged(after, changed);
        }

        for (auto &lhs_val : lhs_pt.get_set_ref()) {
            auto insert_point = after->pt_relation.find(lhs_val);
            if (insert_point == after->pt_relation.end())
                continue;
            changed |= pt_lattice::pointee_union(insert_point->second, rhs_pt);
        }
        propagateIfChanged(after, changed);
    };

    void visit_pt_op(pt::CopyOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);

        auto pt_set = pt_lattice::new_pointee_set();

        for (auto operand : op.getOperands()) {
            auto operand_it = before.find(operand);
            if (operand_it != before.end()) {
                std::ignore = pt_lattice::pointee_union(pt_set, operand_it->second);
            }
        }

        changed |= after->set_var(op.getResult(), pt_set);
        propagateIfChanged(after, changed);
    };

    void visit_pt_op(pt::DereferenceOp &op, const pt_lattice &before, pt_lattice *after) {
        auto changed = after->join(before);

        const auto rhs_pt_it = before.find(op.getPtr());
        if (rhs_pt_it == before.end() || rhs_pt_it->second.is_top()) {

            changed |= after->set_var(op.getResult(), pt_lattice::new_top_set());
            propagateIfChanged(after, changed);
            return;
        }

        auto pointees = pt_lattice::new_pointee_set();
        for (const auto &rhs_val : rhs_pt_it->second.get_set_ref()) {
            auto rhs_it = before.find(rhs_val);
            // We can ignore the change results as they will be resolved by set_var
            if (rhs_it != before.end()) {
                std::ignore = pt_lattice::pointee_union(pointees, rhs_it->second);
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
            auto operand_it = before.find(operand);
            // We can ignore the change results as they will be resolved by set_var
            if (operand_it != before.end()) {
                std::ignore = pt_lattice::pointee_union(pt_set, operand_it->second);
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

    void visitOperation(mlir::Operation *op, const pt_lattice &before, pt_lattice *after) override {
        return llvm::TypeSwitch< mlir::Operation *, void >(op)
            .Case< pt::AddressOp,
                   pt::AllocOp,
                   pt::AssignOp,
                   pt::ConstantOp,
                   pt::CopyOp,
                   pt::DereferenceOp,
                   pt::ValuedConstantOp,
                   pt::UnknownPtrOp >
            ([&](auto &pt_op) { visit_pt_op(pt_op, before, after); })
            .template Case< mlir::UnrealizedConversionCastOp >(
                    [&](auto &cast_op) { visit_unrealized_cast(cast_op, before, after); }
            )
            .Default([&](auto &pt_op) { propagateIfChanged(after, after->join(before)); });
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

        propagateIfChanged(after, after->join(before));
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

        this->propagateIfChanged(lattice, lattice->join(init_state));
    }
};

void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);

void print_analysis_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);

} // potato::analysis
