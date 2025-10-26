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
    llvm::SmallVector< aa_lattice * > get_deref_lats(auto &analysis) const {
        llvm::SmallVector< aa_lattice *> res;
        res.reserve(pointees.size());
        for (const auto &anchor : pointees)
            res.push_back(analysis.template getOrCreate< aa_lattice >(anchor));
        return res;
    }
    void print(llvm::raw_ostream &) const override;

private:
    bool unknown;
    llvm::DenseSet< mlir::LatticeAnchor > pointees;
};

struct aa_analysis : pt_analysis< aa_analysis, aa_lattice > {
    using base = pt_analysis< aa_analysis, aa_lattice >;
    using base::base;

    void register_anchors();
    void set_to_entry_state(aa_lattice *lattice);
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const aa_lattice &l);
} // namespace potato::analysis
