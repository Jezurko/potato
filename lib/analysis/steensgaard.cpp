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

change_result steensgaard::visit_function_model(const function_model &model, fn_interface fn, elem_t res_dummy) {
        auto changed = change_result::NoChange;
        std::vector< mlir_value > from;
        std::vector< mlir_value > deref_from;
        std::vector< mlir_value > copy_to;
        std::vector< mlir_value > assign_to;
        mlir_value realloc_ptr;
        mlir_value realloc_res;
        for (size_t i = 0; i < model.args.size(); i++) {
            auto arg_changed = change_result::NoChange;
            switch(model.args[i]) {
                case arg_effect::none:
                    break;
                case arg_effect::alloc:
                    arg_changed |= new_alloca(fn.getArgument(i));
                    break;
                case arg_effect::realloc_ptr:
                    realloc_ptr = fn.getArgument(i);
                    break;
                case arg_effect::realloc_res:
                    realloc_res = fn.getArgument(i);
                    break;
                case arg_effect::src:
                    from.push_back(fn.getArgument(i));
                    break;
                case arg_effect::copy_trg:
                    copy_to.push_back(fn.getArgument(i));
                    break;
                case arg_effect::deref_src:
                    deref_from.push_back(fn.getArgument(i));
                    break;
                case arg_effect::assign_trg:
                    assign_to.push_back(fn.getArgument(i));
                    break;
                case arg_effect::unknown:
                    arg_changed |= join_var(fn.getArgument(i), new_top_set());
            }
            changed |= arg_changed;
        }

        if (fn.getNumResults() > 0) {
            switch (model.ret) {
                case ret_effect::none:
                    break;
                case ret_effect::alloc: {
                    auto alloca = elem_t::make_alloca(fn.getOperation());
                    changed |= join_var(res_dummy, alloca);
                    break;
                }
                case ret_effect::realloc_res: {
                    auto alloca = elem_t::make_alloca(fn.getOperation());
                    changed |= join_var(res_dummy, alloca);
                    changed |= join_var(realloc_res, lookup(realloc_ptr));
                    break;
                }
                case ret_effect::copy_trg:
                    for (const auto &src : from) {
                        if (auto src_pt = lookup(src); src_pt) {
                            auto trg_changed = join_var(res_dummy, src_pt);
                            changed |= trg_changed;
                        }
                    }
                    for (const auto &src : deref_from) {
                        if (auto src_pt = lookup(src); src_pt) {
                            auto trg_changed = copy_all_pts_into(res_dummy, src_pt);
                            changed |= trg_changed;
                        }
                    }
                    break;
                case ret_effect::assign_trg:
                    if (auto trg_pt = lookup(res_dummy); trg_pt) {
                        if (trg_pt->is_top()) {
                            return set_all_unknown();
                        }
                        for (const auto &src : from) {
                            if (auto src_pt = lookup(src); src_pt) {
                                auto trg_changed = join_all_pointees_with(trg_pt, src_pt);
                                changed |= trg_changed;
                            }
                        }
                    for (const auto &src : deref_from) {
                        if (auto src_pt = lookup(src); src_pt) {
                            auto trg_changed = copy_all_pts_into(*trg_pt, src_pt);
                            changed |= trg_changed;
                        }
                    }
                    }
                    break;
                case ret_effect::unknown:
                    changed |= join_var(res_dummy, new_top_set());
                    break;
            }
        }
        for (const auto &trg : copy_to) {
            for (const auto &src : from) {
                if (auto src_pt = lookup(src); src_pt) {
                    auto trg_changed = join_var(trg, src_pt);
                    changed |= trg_changed;
                }
            }
            for (const auto &src : deref_from) {
                if (auto src_pt = lookup(src); src_pt) {
                    auto trg_elem = elem_t(trg);
                    changed |= copy_all_pts_into(trg_elem, src_pt);
                }
            }
        }
        for (const auto &trg : assign_to) {
            if (auto trg_pt = lookup(trg); trg_pt) {
                if (trg_pt->is_top()) {
                    return set_all_unknown();
                }
                for (const auto &src : from) {
                    if (auto src_pt = lookup(src); src_pt) {
                        auto trg_changed = join_all_pointees_with(trg_pt, src_pt);
                        changed |= trg_changed;
                    }
                }
                for (const auto &src : deref_from) {
                    if (auto src_pt = lookup(src); src_pt) {
                        auto trg_changed = join_all_pointees_with(trg_pt, src_pt);
                        changed |= trg_changed;
                    }
                }
            }
        }
        return changed;
}

fn_info *steensgaard::get_or_create_fn_info(elem_t &elem) {
    mlir_operation *elem_op = elem.elem.has_value() ? elem.elem->operation : nullptr;

    auto fn_info_it = info->fn_infos.find(elem);
    if (fn_info_it != info->fn_infos.end()) {
        return &fn_info_it->second;
    }

    if (!elem.is_func() && !elem_op) {
        return nullptr;
    }

    auto fn = mlir::dyn_cast< fn_interface >(elem_op);
    if (!fn) {
        return nullptr;
    }
    auto args = fn.getArguments();
    mlir_value ret = nullptr;

    if (fn.isExternal()) {
        if (auto model_it = info->models->find(fn.getName()); model_it != info->models->end()) {
            auto res_dummy = new_dummy();
            std::ignore = visit_function_model(model_it->second, fn, new_dummy());
            auto inserted = info->fn_infos.emplace(elem, fn_info(args, res_dummy));
            return &inserted.first->second;
        }
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
    if (lhs_root.is_func() || rhs_root.is_func()) {
        auto lhs_info = get_or_create_fn_info(lhs_root);
        auto rhs_info = get_or_create_fn_info(rhs_root);

        // iff dummy without fn-info no joining
        if (lhs_info && rhs_info) {
            std::optional< mlir_value > last_lhs = std::nullopt;
            std::optional< mlir_value > last_rhs = std::nullopt;

            for (auto [new_arg, old_arg] : llvm::zip_longest(lhs_info->operands, rhs_info->operands)) {
                if (new_arg) {
                    last_lhs = new_arg;
                }
                if (old_arg) {
                    last_rhs = old_arg;
                }
                change |= join_var(last_lhs.value(), *lookup(last_rhs.value()));
            }

            change |= make_union(lhs_info->res, rhs_info->res);
        }
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

    // TODO: check unknown handling
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
