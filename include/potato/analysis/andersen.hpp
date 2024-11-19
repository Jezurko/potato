#pragma once

#include "potato/analysis/lattice.hpp"
#include "potato/analysis/utils.hpp"
#include "potato/dialect/ops.hpp"
#include "potato/util/common.hpp"

#include <memory>
#include <string>

namespace potato::analysis {
struct aa_lattice : mlir_dense_abstract_lattice {
    using mlir_dense_abstract_lattice::AbstractDenseLattice;
    using elem_t = pt_element;
    using pointee_set = lattice_set< elem_t >;
    using relation_t = pt_map< elem_t, lattice_set >;

    std::shared_ptr< pt_map< pt_element, lattice_set > > pt_relation;

    // TODO: remove this
    static unsigned int mem_loc_count;
    unsigned int alloc_count();

    std::optional< std::string > alloc_name = {};
    llvm::StringRef get_alloc_name();

    bool initialized() const { return (bool) pt_relation; }
    void initialize_with(std::shared_ptr< relation_t > &relation) { pt_relation = relation; }

    // TODO: Probably replace most of the following functions with some custom API that doesn't introduce
    //       so many random return values with iterators and stuff

    const pointee_set *lookup(const pt_element &val) const {
        if (!pt_relation)
            return nullptr;
        auto it = pt_relation->find(val);
        if (it == pt_relation->end())
            return nullptr;
        return &it->second;
    }

    const pointee_set *lookup(const mlir_value &val) const {
        return lookup(pt_element(val));
    }

    pointee_set *lookup(const pt_element &val) {
        if (!pt_relation)
            return nullptr;
        auto it = pt_relation->find(val);
        if (it == pt_relation->end())
            return nullptr;
        return &it->second;
    }

    pointee_set *lookup(const mlir_value &val) {
        return lookup(pt_element(val));
    }

    static auto new_symbol(const llvm::StringRef name) {
        return pt_element(name);
    }

    static auto new_pointee_set() {
        return pointee_set();
    }

    static auto new_top_set() {
        return pointee_set::make_top();
    }

    auto join_empty(mlir_value val) {
        auto set = pointee_set();
        auto inserted = pt_relation->insert({{val}, set});
        if (inserted.second) {
            return change_result::Change;
        }
        return change_result::NoChange;
    }

    auto add_constant(mlir_value val) {
        return join_empty(val);
    }

    auto new_alloca(mlir_value val) {
        auto set = pointee_set();
        set.insert({get_alloc_name()});
        auto [it, inserted] = pt_relation->insert({{val}, set});
        if (inserted)
            return change_result::Change;
        else
            return join_var(val, set);

    }

    auto new_var(mlir_value var, const pointee_set& pt_set) {
        assert(pt_relation);
        return pt_relation->insert({{var}, pt_set});
    }

    auto new_var(mlir_value var, mlir_value pointee) {
        pointee_set set{};
        auto pointee_it = pt_relation->find({pointee});
        if (pointee_it == pt_relation->end()) {
            set.insert({get_alloc_name()});
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
            var_pt_set = set;
            return change_result::Change;
        }
        return change_result::NoChange;
    }

    change_result set_var(mlir_value val, mlir_value pointee) {
        auto [var, inserted] = new_var(val, pointee);
        if (inserted)
            return change_result::Change;
        auto &var_pt_set = var->second;
        auto [pointee_var, _] = new_var(pointee, new_pointee_set());
        auto cmp_set = new_pointee_set();
        cmp_set.insert(pointee_var->first);
        if (var_pt_set != cmp_set) {
            var_pt_set = cmp_set;
            return change_result::Change;
        }
        return change_result::NoChange;
    }

    change_result set_var(pt_element elem, const pointee_set &set) {
        auto [var, inserted] = pt_relation->insert({elem, set});
        if (inserted)
            return change_result::Change;
        if (var->second != set) {
            var->second = set;
            return change_result::Change;
        }
        return change_result::NoChange;
    }

    change_result join_var(mlir_value val, mlir_value trg) {
        auto val_pt  = lookup(val);
        if (!val_pt) {
            return set_var(val, pointee_set({trg}));
        }
        return val_pt->join(pointee_set({trg}));
    }

    change_result join_var(mlir_value val, const pointee_set &set) {
        auto val_pt  = lookup(val);
        if (!val_pt) {
            return set_var(val, set);
        }
        return val_pt->join(set);
    }

    change_result join_var(pt_element elem, const pointee_set &set) {
        auto val_pt  = lookup(elem);
        if (!val_pt) {
            return set_var(elem, set);
        }
        return val_pt->join(set);
    }

    change_result join_var(mlir_value val, const pointee_set *set) {
        auto val_pt  = lookup(val);
        if (!val_pt) {
            return set_var(val, *set);
        }
        return val_pt->join(*set);
    }

    change_result join_var(pt_element elem, const pointee_set *set) {
        auto val_pt  = lookup(elem);
        if (!val_pt) {
            return set_var(elem, *set);
        }
        return val_pt->join(*set);
    }

    // for each p in to lookup pts(p) and join it with from
    change_result join_all_pointees_with(pointee_set *to, const pointee_set *from) {
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

    // for each p in from do pts(to) join_with pts(p)
    change_result copy_all_pts_into(pt_element to, const pointee_set *from) {
        auto changed = change_result::NoChange;
        std::vector< const pointee_set * > to_join;

        for (const auto &val : from->get_set_ref()) {
            auto val_pt = lookup(val);
            if (val_pt) {
                to_join.push_back(val_pt);
            }
        }

        // make sure `to` is in the lattice
        changed |= join_var(to, pointee_set());

        for (auto *join : to_join) {
            changed |= join_var(to, join);
        }

        return changed;
    }

    change_result set_all_unknown() {
        auto changed = change_result::NoChange;
        for (auto &[_, pt_set] : *pt_relation) {
            changed |= pt_set.set_top();
        }
        return changed;
    }

    change_result merge(const aa_lattice &rhs) {
        if (pt_relation && !rhs.pt_relation)
            return change_result::NoChange;
        if (pt_relation && rhs.pt_relation == pt_relation) {
            return change_result::NoChange;
        }
        if (pt_relation && rhs.pt_relation) {
            llvm::errs() << "Merging two different relations.\n";
            change_result res = change_result::NoChange;
            for (const auto &[key, rhs_value] : *rhs.pt_relation) {
                auto &lhs_value = (*pt_relation)[key];
                res |= lhs_value.join(rhs_value);
            }
            return res;
        }
        if (rhs.pt_relation) {
            pt_relation = rhs.pt_relation;
            return change_result::Change;
        }
        assert(false);
        return change_result::NoChange;
    }

    change_result intersect(const aa_lattice &rhs) {
        if (rhs.pt_relation == pt_relation) {
            return change_result::NoChange;
        }
        change_result res = change_result::NoChange;
        for (const auto &[key, rhs_value] : *rhs.pt_relation) {
            // non-existent entry would be considered top, so creating a new entry
            // and intersecting it will create the correct value
            auto &lhs_value = (*pt_relation)[key];
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

    void print(llvm::raw_ostream &os) const override;

    static void add_dependencies(mlir::Operation *op, mlir_dense_dfa< aa_lattice > *analysis, ppoint point, auto get_or_create) {
        for (auto arg : op->getOperands()) {
            aa_lattice *arg_state;
            if (auto def_op = arg.getDefiningOp()) {
                arg_state = get_or_create(arg.getDefiningOp());
            } else {
                arg_state = get_or_create(arg.getParentBlock());
            }
            arg_state->addDependency(point, analysis);
        }
    }

    constexpr static bool propagate_assign() { return true; }
    constexpr static bool propagate_call_arg_zip() { return true; }

    alias_res alias(auto lhs, auto rhs) const {
        const auto lhs_pt = lookup(lhs);
        const auto rhs_pt = lookup(rhs);
        // If we do not know at least one of the arguments we can not deduce any aliasing information
        // TODO: can this happen with correct usage? Should we emit a warning?
        if (!lhs_pt || !rhs_pt)
            return alias_res(alias_kind::MayAlias);

        if (sets_intersect(lhs_pt->get_set_ref(), rhs_pt->get_set_ref())) {
            if (lhs_pt->is_single_target() && rhs_pt->is_single_target()) {
                for (const auto &member : lhs_pt->get_set_ref()) {
                    // for dynamically allocated values we can not return MustAlias.
                    if (std::holds_alternative< mlir_value >(member.id))
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
