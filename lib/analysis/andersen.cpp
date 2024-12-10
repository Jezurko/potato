#include "potato/analysis/andersen.hpp"
namespace potato::analysis {

const aa_lattice::pointee_set *aa_lattice::lookup(const elem_t &val) const {
    if (!info)
        return nullptr;
    auto it = pt_relation().find(val);
    if (it == pt_relation().end())
        return nullptr;
    return &it->second;
}

aa_lattice::pointee_set *aa_lattice::lookup(const elem_t &val) {
    if (!info)
        return nullptr;
    auto it = pt_relation().find(val);
    if (it == pt_relation().end())
        return nullptr;
    return &it->second;
}

change_result aa_lattice::join_empty(mlir_value val) {
    auto set = pointee_set();
    auto inserted = pt_relation().insert({{val}, set});
    if (inserted.second) {
        return change_result::Change;
    }
    return change_result::NoChange;
}

change_result aa_lattice::new_alloca(mlir_value val, mlir_operation *alloc) {
    auto set = pointee_set();
    set.insert(elem_t::make_alloca(alloc));
    auto [it, inserted] = pt_relation().insert({{val}, set});
    if (inserted)
        return change_result::Change;
    else
        return join_var(val, set);
}

change_result aa_lattice::new_alloca(mlir_value val) {
    return new_alloca(val, val.getDefiningOp());
}

change_result aa_lattice::deref_alloca(mlir_value val, mlir_operation *op) {
    if (auto pt = lookup(val)) {
        auto set = pointee_set();
        auto deref_alloc = elem_t::make_alloca(op, val);
        set.insert(deref_alloc);
        return join_all_pointees_with(pt, &set);
    }
    return change_result::NoChange;

}

void aa_lattice::add_argc(mlir_value value, mlir_operation *op) {
    auto str = elem_t::make_alloca(op);
    auto str_ptr = elem_t::make_alloca(op, value);
    std::ignore = join_var(str_ptr, str);
    std::ignore = join_var(value, str_ptr);
}

change_result aa_lattice::join_var(mlir_value val, mlir_value trg) {
    auto val_pt  = lookup(val);
    if (!val_pt) {
        return set_var(val, elem_t::make_alloca(trg.getDefiningOp()));
    }
    return val_pt->join(pointee_set(elem_t::make_alloca(trg.getDefiningOp())));
}

change_result aa_lattice::join_var(mlir_value val, const pointee_set &set) {
    auto val_pt  = lookup(val);
    if (!val_pt) {
        return set_var(val, set);
    }
    return val_pt->join(set);
}

change_result aa_lattice::join_var(elem_t elem, const pointee_set &set) {
    auto val_pt  = lookup(elem);
    if (!val_pt) {
        return set_var(elem, set);
    }
    return val_pt->join(set);
}

change_result aa_lattice::join_var(mlir_value val, const pointee_set *set) {
    auto val_pt  = lookup(val);
    if (!val_pt) {
        return set_var(val, *set);
    }
    return val_pt->join(*set);
}

change_result aa_lattice::join_var(elem_t elem, const pointee_set *set) {
    auto val_pt  = lookup(elem);
    if (!val_pt) {
        return set_var(elem, *set);
    }
    return val_pt->join(*set);
}

change_result aa_lattice::join_all_pointees_with(pointee_set *to, const pointee_set *from) {
    auto changed = change_result::NoChange;
    std::vector< const elem_t * > to_update;
    for (auto &val : to->get_set_ref()) {
        to_update.push_back(&val);
    }
    // This has to be done so that we don't change the set we are iterating over under our hands
    for (auto &key : to_update) {
        changed |= join_var(*key, from);
    }

    return changed;
}

change_result aa_lattice::copy_all_pts_into(elem_t to, const pointee_set *from) {
    auto changed = change_result::NoChange;
    std::vector< const pointee_set * > to_join;

    for (const auto &val : from->get_set_ref()) {
        auto val_pt = lookup(val);
        if (val_pt) {
            to_join.push_back(val_pt);
        }
    }

    // make sure `to` is in the lattice
    if (to_join.empty())
        changed |= join_var(to, pointee_set());

    for (auto *join : to_join) {
        changed |= join_var(to, join);
    }

    return changed;
}

change_result aa_lattice::copy_all_pts_into(const pointee_set *to, const pointee_set *from) {
    auto changed = change_result::NoChange;
    for (auto member : to->get_set_ref()) {
        changed |= copy_all_pts_into(member, from);
    }
    return changed;
}

change_result aa_lattice::merge(const aa_lattice &rhs) {
    if (info && !rhs.info)
        return change_result::NoChange;
    if (info && rhs.info == info) {
        return change_result::NoChange;
    }
    if (info && rhs.info) {
        llvm::errs() << "Merging two different relations.\n";
        change_result res = change_result::NoChange;
        for (const auto &[key, rhs_value] : rhs.pt_relation()) {
            auto &lhs_value = pt_relation()[key];
            res |= lhs_value.join(rhs_value);
        }
        return res;
    }
    if (rhs.info) {
        info = rhs.info;
        return change_result::Change;
    }
    assert(false);
    return change_result::NoChange;
}

change_result aa_lattice::intersect(const aa_lattice &rhs) {
    if (rhs.info == info) {
        return change_result::NoChange;
    }
    change_result res = change_result::NoChange;
    for (const auto &[key, rhs_value] : rhs.pt_relation()) {
        // non-existent entry would be considered top, so creating a new entry
        // and intersecting it will create the correct value
        auto &lhs_value = pt_relation()[key];
        res |= lhs_value.meet(rhs_value);
    }
    return res;
}

void aa_lattice::print(llvm::raw_ostream &os) const {
    auto sep = "";
    for (const auto &[key, vals] : pt_relation()) {
        os << sep << key << " -> " << vals;
        sep = "\n";
    }
    os << "\n";
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const potato::analysis::aa_lattice &l) { l.print(os); return os; }
} // namespace potato::analysis

