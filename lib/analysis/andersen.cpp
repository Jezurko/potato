#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/SetOperations.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/andersen.hpp"
#include "potato/analysis/pt.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis {
    mlir_loc mem_loc_anchor::getLoc() const { return getValue().first->getLoc(); }

    void mem_loc_anchor::print(llvm::raw_ostream &os) const {
        os << "mem_alloc at: " << getLoc();
        if (getValue().second != 0)
            os << "with unique id " << getValue().second;
    }

    mlir_loc named_val_anchor::getLoc() const { return getValue()->getLoc(); }

    void named_val_anchor::print(llvm::raw_ostream &os) const {
        if (auto symbol = mlir::dyn_cast< symbol_iface >(getValue()))
            os << symbol.getName();
        else
            os << "named_val: " << *getValue();
    }

    change_result aa_lattice::set_unknown() {
        if (unknown)
            return change_result::NoChange;
        unknown = true;
        return change_result::NoChange;
    }

    change_result aa_lattice::join(const aa_lattice &rhs) {
        if (unknown)
            return change_result::NoChange;

        if (rhs.unknown)
            return set_unknown();

        return llvm::set_union(pointees, rhs.pointees) ?
            change_result::Change : change_result::NoChange;
    }

    change_result aa_lattice::insert(lattice_anchor anchor) {
        return pointees.insert(anchor).second ?
            change_result::Change : change_result::NoChange;
    }

    void aa_lattice::print(llvm::raw_ostream &os) const {
        os << getAnchor() << " -> ";

        if (unknown) {
            os << "UNKNOWN\n";
            return;
        }

        std::string sep = "";
        os << "{";
        for (const auto &pointee : pointees)
            os << sep << pointee;
        os << "}\n";
    }

    void aa_analysis::register_anchors() {}
    void aa_analysis::set_to_entry_state(aa_lattice *lattice) { assert(false); }
} // namespace potato::analysis

