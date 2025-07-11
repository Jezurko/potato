#pragma once

#include "potato/analysis/lattice.hpp"
#include "potato/analysis/utils.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis {

struct aa_lattice : mlir_dense_abstract_lattice {
    using mlir_dense_abstract_lattice::AbstractDenseLattice;
    using elem_t = pt_element;
    using pointee_set = lattice_set< elem_t >;
    using relation_t = pt_map< elem_t, lattice_set >;
    struct aa_info {
        pt_map< elem_t, lattice_set > pt_relation;
        bool all_unknown;
    };
    using info_t = aa_info;

    aa_info *info = nullptr;

    bool initialized() const { return (bool)info; }
    void initialize_with(info_t *new_info) { info = new_info; }

    relation_t &pt_relation() const { return info->pt_relation; }

    const pointee_set *lookup(const elem_t &val) const;
    pointee_set *lookup(const elem_t &val);
    pointee_set *lookup(const mlir_value &val) { return lookup(elem_t(val)); }

    static elem_t new_func(mlir_operation *op) { return elem_t::make_func(op); }
    static elem_t new_named_var(mlir_operation *op) { return elem_t::make_named_var(op); }
    static pointee_set new_pointee_set() { return pointee_set(); }
    static pointee_set new_top_set() { return pointee_set::make_top(); }

    change_result join_empty(mlir_value val);

    change_result add_constant(mlir_value val) { return join_empty(val); }
    change_result new_alloca(mlir_value val, mlir_operation *alloc);
    change_result new_alloca(mlir_value val);
    change_result deref_alloca(mlir_value val, mlir_operation *op);
    void add_argc(mlir_value val, mlir_operation *op);
    auto new_var(mlir_value var, const pointee_set& pt_set) {
        return pt_relation().insert({{var}, pt_set});
    }

    auto new_var(mlir_value var, mlir_value pointee) {
        pointee_set set{};
        auto pointee_it = pt_relation().find({pointee});
        if (pointee_it == pt_relation().end()) {
            set.insert(elem_t::make_alloca(var.getDefiningOp()));
        } else {
            set.insert(pointee_it->first);
        }
        return new_var(var, set);
    }

    change_result set_var(mlir_value val, const pointee_set &set);
    change_result set_var(mlir_value val, mlir_value pointee);
    change_result set_var(elem_t elem, const pointee_set &set);

    change_result join_var(mlir_value val, mlir_value trg);
    change_result join_var(mlir_value val, const pointee_set &set);
    change_result join_var(elem_t elem, const pointee_set &set);
    change_result join_var(mlir_value val, const pointee_set *set);
    change_result join_var(elem_t elem, const pointee_set *set);

    // for each p in to lookup pts(p) and join it with from
    change_result join_all_pointees_with(pointee_set *to, const pointee_set *from);
    // for each p in from do pts(to) join_with pts(p)
    change_result copy_all_pts_into(elem_t to, const pointee_set *from);
    change_result copy_all_pts_into(const pointee_set *to, const pointee_set *from);

    static void propagate_members_changed(const pointee_set *set, auto get_or_create, auto propagate) {
        for (const auto &member : set->get_set_ref()) {
            if (member.is_named_var() || member.is_alloca()) {
                auto state = get_or_create(member.operation);
                propagate(state, change_result::Change);
            }
            if (member.is_var()) {
                auto state = get_or_create(member.val.getDefiningOp());
                propagate(state, change_result::Change);
            }
        }
    }

    static void depend_on_members(pointee_set *set, auto add_dep) {
        for (auto &member : set->get_set_ref()) {
            if (member.is_named_var() || member.is_alloca()) {
                add_dep(member.operation);
            }
            if (member.is_var()) {
                if (auto op = member.val.getDefiningOp())
                    add_dep(op);
                else
                    add_dep(member.val.getParentBlock());
            }
        }
    }

    change_result resolve_fptr_call(
        mlir_value val,
        mlir::CallOpInterface call,
        auto get_or_create,
        auto add_dep,
        auto propagate,
        auto analysis
    ) {
        auto changed = change_result::NoChange;
        auto fptr_pt = lookup(val);
        if (!fptr_pt)
            return changed;

        if (fptr_pt->is_top()) {
            return set_all_unknown();
        }

        for (const auto &fun : fptr_pt->get_set_ref()) {
            if(!fun.is_func()) {
                continue;
            }
            auto fn = mlir::dyn_cast< mlir::FunctionOpInterface >(fun.operation);
            if (fn.isExternal()) {
                continue;
            } else {
                // FIXME: this is almost copy paste from pt.hpp. Can we unify it?
                auto &callee_entry = fn->getRegion(0).front();
                auto callee_args   = callee_entry.getArguments();

                mlir_value last_call_arg;
                mlir_value last_callee_arg;
                for (const auto &[callee_arg, caller_arg] :
                     llvm::zip_longest(callee_args, call.getArgOperands()))
                {
                    if (caller_arg) {
                        last_call_arg = caller_arg.value();
                    }
                    if (callee_arg) {
                        last_callee_arg = callee_arg.value();
                    }
                    if (auto arg_pt = lookup(last_call_arg))
                        changed |= join_var(last_callee_arg, arg_pt);
                }
                propagate(get_or_create(&callee_entry), changed);

                auto handle_return = [&](mlir_operation *op) {
                        if (op->hasTrait< mlir::OpTrait::ReturnLike >()) {
                            for (size_t i = 0; i < op->getNumOperands(); i++) {
                                auto res_arg = op->getOperand(i);
                                if (auto res_pt = lookup(res_arg)) {
                                    changed |= join_var(call->getResult(i), res_pt);
                                }
                                if (auto op = res_arg.getDefiningOp())
                                    add_dep(op);
                                else
                                    add_dep(res_arg.getParentBlock());
                            }
                        }
                    };
                if (call->getNumResults() > 0)
                    fn.getFunctionBody().walk(handle_return);
            }
        }
        return changed;
    }

    change_result set_all_unknown() {
        if (!info->all_unknown)
            info->all_unknown = true;
        return change_result::Change;
    }

    change_result merge(const aa_lattice &rhs);
    change_result intersect(const aa_lattice &rhs);

    change_result join(const mlir_dense_abstract_lattice &rhs) override {
        return this->merge(*static_cast< const aa_lattice *>(&rhs));
    };

    change_result meet(const mlir_dense_abstract_lattice &rhs) override {
        return this->intersect(*static_cast< const aa_lattice *>(&rhs));
    };

    bool is_all_unknown() const { return info->all_unknown; }

    void print(llvm::raw_ostream &os) const override;

    static void add_dependencies(
            mlir::Operation *op,
            mlir_dense_dfa< aa_lattice > *analysis,
            ppoint point,
            auto get_or_create
    ) {
        for (auto arg : op->getOperands()) {
            aa_lattice *arg_state;
            if (auto def_op = arg.getDefiningOp()) {
                arg_state = get_or_create(def_op);
            } else {
                arg_state = get_or_create(arg.getParentBlock());
            }
            arg_state->addDependency(point, analysis);
        }
    }

    constexpr static bool propagate_assign() { return true; }
    constexpr static bool propagate_call_arg_zip() { return true; }

    alias_res alias(auto lhs, auto rhs) const {
        if (info->all_unknown) {
            return alias_res(alias_kind::MayAlias);
        }
        const auto lhs_pt = lookup(lhs);
        const auto rhs_pt = lookup(rhs);
        // If we do not know at least one of the arguments we can not deduce any aliasing information
        // TODO: can this happen with correct usage? Should we emit a warning?
        if (!lhs_pt || !rhs_pt) {
            assert(false);
            return alias_res(alias_kind::MayAlias);
        }

        if (lhs_pt->is_top() || rhs_pt->is_top()) {
            return alias_res(alias_kind::MayAlias);
        }

        if (sets_intersect(lhs_pt->get_set_ref(), rhs_pt->get_set_ref())) {
            if (lhs_pt->is_single_target() && rhs_pt->is_single_target()) {
                for (const auto &member : lhs_pt->get_set_ref()) {
                    // for dynamically allocated values we can not return MustAlias.
                    if (!member.is_alloca())
                        return alias_res(alias_kind::MustAlias);
                }
            }
            return alias_res(alias_kind::MayAlias);
        }

        return alias_res(alias_kind::NoAlias);
    }
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const aa_lattice &l);
} // namespace potato::analysis
