#include "potato/analysis/steensgaard.hpp"
namespace potato::analysis {

void stg_elem::print(llvm::raw_ostream &os) const {
    if (elem) {
        elem->print(os);
        return;
    }
    if (dummy_id) {
        os << "dummy_" << *dummy_id;
        return;
    }
    if (is_top()) {
        os << "top";
        return;
    }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const stg_elem &e) {
    e.print(os);
    return os;
}

steensgaard::elem_t *steensgaard::lookup(const elem_t &val) {
    auto val_rep = sets().find(val);
    auto trg_it = mapping().find(val_rep);
    if (trg_it == mapping().end()) {
        return &mapping().emplace(val_rep, new_dummy()).first->second;
    }
    return &trg_it->second;
}

steensgaard::elem_t steensgaard::new_dummy() {
    auto dummy = elem_t(info->dummy_count++);
    sets().insert(dummy);
    return dummy;
}

change_result steensgaard::new_alloca(mlir_value val) {
    auto alloca = elem_t::make_alloca(val.getDefiningOp());
    return join_var(val, alloca);
}

change_result steensgaard::join_all_pointees_with(elem_t *to, const elem_t *from) {
    auto to_rep = sets().find(*to);
    auto to_trg_it = mapping().find(to_rep);
    if (to_trg_it == mapping().end()) {
        mapping().emplace(to_rep, *from);
        return change_result::Change;
    }
    return make_union(to_trg_it->second, *from);
}

change_result steensgaard::copy_all_pts_into(elem_t &&to, const elem_t *from) {
    auto from_rep = sets().find(*from);
    auto to_rep = sets().find(to);
    auto to_trg_it = mapping().find(to_rep);
    auto from_trg_it = mapping().find(from_rep);

    if (from_trg_it == mapping().end()) {
        if (to_trg_it == mapping().end()) {
            // tie targets with a dummy
            auto dummy = new_dummy();
            mapping().emplace(to_rep, dummy);
            mapping().emplace(from_rep, dummy);
        } else {
            // tie targets together
            mapping().emplace(from_rep, to_trg_it->second);
        }
        return change_result::Change;
    }

    auto &from_trg = from_trg_it->second;

    if (to_trg_it == mapping().end()) {
        mapping().emplace(to_rep, from_trg);
        return change_result::Change;
    }

    return make_union(to_trg_it->second, from_trg);
}

change_result steensgaard::make_union(elem_t lhs, elem_t rhs) {
    auto lhs_root = sets().find(lhs);
    auto rhs_root = sets().find(rhs);
    if (lhs_root == rhs_root) {
        return change_result::NoChange;
    }

    auto lhs_trg = mapping().find(lhs_root);
    auto rhs_trg = mapping().find(rhs_root);

    auto new_root = sets().set_union(lhs, rhs);

    if (lhs_trg == mapping().end() && rhs_trg == mapping().end()) {
        return change_result::Change;
    }

    // merge outgoing edges
    if (lhs_trg == mapping().end()) {
        if (lhs_root == new_root) {
            mapping().emplace(new_root, rhs_trg->second);
            mapping().erase(rhs_trg);
        }
        return change_result::Change;
    }
    if (rhs_trg == mapping().end()) {
        if (rhs_root == new_root) {
            mapping().emplace(new_root, lhs_trg->second);
            mapping().erase(lhs_trg);
        }
        return change_result::Change;
    }

    if (lhs_trg->second.is_unknown()) {
        mapping()[rhs_root] = lhs_trg->second;
        return change_result::Change;
    }

    if (rhs_trg->second.is_unknown()) {
        mapping()[lhs_root] = rhs_trg->second;
    }

    // if both have outgoing edge, unify the targets and remove the redundant edge
    std::ignore = make_union(lhs_trg->second, rhs_trg->second);
    if (new_root == lhs_root) {
        mapping().erase(rhs_trg);
    } else {
        mapping().erase(lhs_trg);
    }

    return change_result::Change;
}

change_result steensgaard::join_var(const elem_t &ptr, const elem_t &new_trg) {
    auto ptr_rep = sets().find(ptr);
    auto new_trg_rep = sets().find(new_trg);

    auto ptr_trg = mapping().find(ptr_rep);
    if (ptr_trg == mapping().end()) {
        mapping().emplace(ptr_rep, new_trg_rep);
        return change_result::Change;
    }
    auto ptr_trg_rep = sets().find(ptr_trg->second);
    if (ptr_trg_rep.is_unknown()) {
        return change_result::NoChange;
    }
    if (new_trg_rep.is_unknown()) {
        mapping()[ptr_rep] = new_trg_rep;
        return change_result::Change;
    }
    return make_union(ptr_trg->second, new_trg_rep);
}

change_result steensgaard::set_all_unknown() {
    auto &unknown = all_unknown();
    if (unknown) {
        return change_result::NoChange;
    }

    unknown = true;
    return change_result::Change;
}

change_result steensgaard::merge(const steensgaard &rhs) {
    if (info && !rhs.info)
        return change_result::NoChange;
    if (info && rhs.info == info) {
        return change_result::NoChange;
    }
    if (info && rhs.info) {
        llvm::errs() << "Merging two different relations.\n";
        //TODO
    }
    if (rhs.info) {
        info = rhs.info;
        return change_result::Change;
    }
    return change_result::NoChange;
};

void steensgaard::print(llvm::raw_ostream &os) const {
    for (const auto &[src, trg] : mapping()) {
        src.print(os);
        os << " -> {";
        auto trg_rep = sets().find(trg);
        auto sep = "";
        for (auto child : sets().children.at(trg_rep)) {
            os << sep;
            child.print(os);
            sep = ", ";
        }
        os << "}\n";
    }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const steensgaard &l) {
    l.print(os);
    return os;
}
} // namespace potato::analysis
