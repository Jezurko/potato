#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/BitVector.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/IndexedMap.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/pt.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis {

struct anchor_index {
    anchor_index(llvm::SmallVector< lattice_anchor > anchor_list);

    size_t anchor_to_idx(lattice_anchor anchor) const { return anchor_to_index.at(anchor); }
    lattice_anchor idx_to_anchor(size_t idx) const { return anchor_list[idx]; }
    size_t get_size() const { return anchor_list.size(); }

private:
    llvm::SmallVector< lattice_anchor > anchor_list;
    llvm::DenseMap< lattice_anchor, size_t > anchor_to_index;
};

struct aa_bv_lattice : pt_lattice_base< aa_bv_lattice > {
    using base = pt_lattice_base< aa_bv_lattice >;

    aa_bv_lattice(lattice_anchor anchor) : pt_lattice_base(anchor), unknown(false) {};
    aa_bv_lattice(mlir_value value) : pt_lattice_base(value), unknown(false) {};

    bool is_unknown() const { return unknown; }
    change_result set_unknown();
    change_result join(const aa_bv_lattice &);
    change_result insert(lattice_anchor);
    alias_res alias_impl(aa_bv_lattice *rhs);
    const llvm::DenseSet< lattice_anchor > &get_pointees() const;
    void print(llvm::raw_ostream &) const override;

private:
    bool unknown = false;
    llvm::BitVector pointees;
    const anchor_index *index;
};

struct aa_bv_analysis : pt_analysis< aa_bv_analysis, aa_bv_lattice > {
    using base = pt_analysis< aa_bv_analysis, aa_bv_lattice >;
    using base::base;

    void register_anchors();
    void set_to_entry_state_impl(aa_bv_lattice *lattice);

    logical_result initialize(mlir_operation *root) override;

private:
    anchor_index index;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const aa_bv_lattice &l);
} // namespace potato::analysis
