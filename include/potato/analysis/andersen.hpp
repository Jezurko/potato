#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/DenseSet.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/pt.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis {

struct aa_lattice : pt_lattice_base< aa_lattice > {

    aa_lattice(lattice_anchor anchor) : pt_lattice_base(anchor), unknown(false) {};
    aa_lattice(mlir_value value) : pt_lattice_base(value), unknown(false) {};

    change_result set_unknown();
    change_result join(const aa_lattice &);
    change_result insert(lattice_anchor);
    alias_res alias_impl(aa_lattice *rhs);
    const llvm::DenseSet< lattice_anchor > &get_pointees() const { return pointees; }
    void print(llvm::raw_ostream &) const override;

private:
    bool unknown = false;
    llvm::DenseSet< lattice_anchor > pointees;
};

struct aa_analysis : pt_analysis< aa_analysis, aa_lattice > {
    using base = pt_analysis< aa_analysis, aa_lattice >;
    using base::base;

    void register_anchors();
    void set_to_entry_state(aa_lattice *lattice);
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const aa_lattice &l);
} // namespace potato::analysis
