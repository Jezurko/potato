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

change_result steensgaard::new_alloca(mlir_value val, mlir_operation *op) {
    auto alloca = elem_t::make_alloca(op);
    return join_var(val, alloca);
}

change_result steensgaard::new_alloca(mlir_value val) {
    return new_alloca(val, val.getDefiningOp());
}

change_result steensgaard::deref_alloca(mlir_value val, mlir_operation *op) {
    auto deref_alloc = elem_t::make_alloca(op, val);
    auto pt = lookup(val);
    return join_all_pointees_with(pt, &deref_alloc);
}

void steensgaard::add_argc(mlir_value value, mlir_operation *op) {
    auto str = elem_t::make_alloca(op);
    auto str_ptr = elem_t::make_alloca(op, value);
    std::ignore = join_var(str_ptr, str);
    std::ignore = join_var(value, str_ptr);
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

change_result steensgaard::copy_all_pts_into(const elem_t &to, const elem_t *from) {
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

change_result steensgaard::copy_all_pts_into(const elem_t *to, const elem_t *from) {
    return copy_all_pts_into(*to, from);
}
fn_info *steensgaard::get_or_create_fn_info(elem_t &elem) {
    mlir_operation *elem_op = elem.elem.has_value() ? elem.elem->operation : nullptr;

    auto fn_info_it = info->fn_infos.find(elem);
    if (fn_info_it != info->fn_infos.end()) {
        return &fn_info_it->second;
    }

    if (!elem.is_func() && !(elem.is_dummy() && elem_op)) {
        return nullptr;
    }

    auto fn = mlir::dyn_cast< func_iface >(elem_op);
    if (!fn) {
        return nullptr;
    }
    mlir_value ret = nullptr;

    if (fn.isExternal()) {
        assert(false);
        return nullptr;
    }

    elem_t *pt = nullptr;
    if (fn.getNumResults() > 0) {
        fn->walk([&](mlir_operation *op) {
            if (op->hasTrait< mlir::OpTrait::ReturnLike >()) {
                // merges also multiple returns of a function
                // but we can assume that every function has a single return value for now
                for (auto operand : op->getOperands()) {
                    auto trg = lookup(operand);
                    if (pt) {
                        // FIXME: ignore
                        std::ignore = make_union(*pt, *trg);
                    } else {
                        pt = trg;
                    }
                    if (!ret) {
                        ret = operand;
                    }
                }
            }
        });
    }
    std::vector< elem_t > args{};
    for (const auto &arg : fn.getArguments()) {
        args.push_back(arg);
    }
    auto inserted = info->fn_infos.emplace(elem, fn_info(args, pt ? *pt : new_dummy()));
    return &inserted.first->second;
}

change_result steensgaard::make_union(elem_t lhs, elem_t rhs) {
    auto lhs_root = sets().find(lhs);
    auto rhs_root = sets().find(rhs);
    if (lhs_root == rhs_root) {
        return change_result::NoChange;
    }

    auto change = change_result::NoChange;

    auto lhs_trg = mapping().find(lhs_root);
    auto rhs_trg = mapping().find(rhs_root);

    auto new_root = sets().set_union(lhs, rhs);

    auto make_fn_union = [&]() {
        if (lhs_root.is_func() || rhs_root.is_func()) {
            auto lhs_info = get_or_create_fn_info(lhs_root);
            auto rhs_info = get_or_create_fn_info(rhs_root);

            // iff dummy without fn-info no joining
            if (lhs_info && rhs_info) {
                stg_elem last_lhs{};
                stg_elem last_rhs{};

                for (auto [new_arg, old_arg] : llvm::zip_longest(lhs_info->operands, rhs_info->operands)) {
                    if (new_arg) {
                        last_lhs = new_arg.value();
                    }
                    if (old_arg) {
                        last_rhs = old_arg.value();
                    }
                    change |= join_var(sets().find(last_lhs), *lookup(last_rhs));
                }
                change |= make_union(lhs_info->res, rhs_info->res);
            }
        }
    };

    if (lhs_trg == mapping().end() && rhs_trg == mapping().end()) {
        make_fn_union();
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

    if (sets().find(lhs_trg->second).is_unknown()) {
        mapping()[rhs_root] = lhs_trg->second;
    }

    if (sets().find(rhs_trg->second).is_unknown()) {
        mapping()[lhs_root] = rhs_trg->second;
    }

    // if both have outgoing edge, unify the targets and remove the redundant edge
    std::ignore = make_union(lhs_trg->second, rhs_trg->second);
    make_fn_union();

    if (new_root == lhs_root) {
        mapping().erase(rhs_trg);
    } else {
        mapping().erase(lhs_trg);
    }

    return change;
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
    if (new_trg_rep == ptr_trg_rep) {
        return change_result::NoChange;
    }
    if (ptr_trg_rep.is_unknown()) {
        return change_result::NoChange;
    }
    if (new_trg_rep.is_unknown()) {
        mapping()[ptr_rep] = new_trg_rep;
        return change_result::Change;
    }
    return make_union(ptr_trg_rep, new_trg_rep);
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
        assert(false);
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
