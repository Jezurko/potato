#pragma once

#include "potato/analysis/lattice.hpp"
#include "potato/analysis/utils.hpp"
#include "potato/dialect/ops.hpp"
#include "potato/util/common.hpp"

#include <string>

namespace potato::analysis {
struct fsa_lattice
{
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

    const pointee_set *lookup(const pt_element &val) const {
        auto it = pt_relation.find(val);
        if (it == pt_relation.end())
            return nullptr;
        return &it->second;
    }

    const pointee_set *lookup(const mlir_value &val) const {
        return lookup({ val, "" });
    }

    pointee_set *lookup(const pt_element &val) {
        auto it = pt_relation.find(val);
        if (it == pt_relation.end())
            return nullptr;
        return &it->second;
    }

    pointee_set *lookup(const mlir_value &val) {
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
        auto pointee_it = pt_relation.find({ pointee, "" });
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
        auto [var, inserted] = pt_relation.insert({elem, set});
        if (inserted)
            return change_result::Change;
        if (var->second != set) {
            var->second = set;
            return change_result::Change;
        }
        return change_result::NoChange;
    }

    change_result join_var(mlir_value val, pointee_set &&set) {
        auto val_pt  = lookup(val);
        if (!val_pt) {
            return set_var(val, set);
        }
        return val_pt->join(set);
    }

    change_result join_var(mlir_value val, const pointee_set &set) {
        auto val_pt  = lookup(val);
        if (!val_pt) {
            return set_var(val, set);
        }
        return val_pt->join(set);
    }

    change_result set_all_unknown() {
        auto changed = change_result::NoChange;
        for (auto &[_, pt_set] : pt_relation) {
            changed |= pt_set.set_top();
        }
        return changed;
    }

    change_result join(const fsa_lattice &rhs) {
        change_result res = change_result::NoChange;
        for (const auto &[key, rhs_value] : rhs.pt_relation) {
            auto &lhs_value = pt_relation[key];
            res |= lhs_value.join(rhs_value);
        }
        return res;
    }

    change_result meet(const fsa_lattice &rhs) {
        change_result res = change_result::NoChange;
        for (const auto &[key, rhs_value] : rhs.pt_relation) {
            // non-existent entry would be considered top, so creating a new entry
            // and intersecting it will create the correct value
            auto &lhs_value = pt_relation[key];
            res |= lhs_value.meet(rhs_value);
        }
        return res;
    }

    void print(llvm::raw_ostream &os) const;

    alias_res alias(auto lhs, auto rhs) const {
        const auto lhs_it = find(lhs);
        const auto rhs_it = find(rhs);
        // If we do not know at least one of the arguments we can not deduce any aliasing information
        // TODO: can this happen with correct usage? Should we emit a warning?
        if (lhs_it == pt_relation.end() || rhs_it() == pt_relation.end())
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
} // namespace potato::analysis
