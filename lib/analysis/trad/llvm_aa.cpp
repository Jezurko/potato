#include "potato/analysis/trad/llvm_aa.hpp"

namespace potato::analysis::trad {

    change_result llaa_lattice::join(const mlir::dataflow::AbstractDenseLattice &rhs) {
        change_result res = change_result::NoChange;
        for (const auto &[key, rhs_value] : static_cast< const llaa_lattice *>(&rhs)->pt_relation) {
            auto &lhs_value = pt_relation[key];
            res |= lhs_value.join(rhs_value);
        }
        return res;
    }

    change_result llaa_lattice::meet(const mlir::dataflow::AbstractDenseLattice &rhs) {
        change_result res = change_result::NoChange;
        for (const auto &[key, rhs_value] : static_cast< const llaa_lattice *>(&rhs)->pt_relation) {
            // non-existent entry would be considered top, so creating a new entry
            // and intersecting it will create the correct value
            auto &lhs_value = pt_relation[key];
            res |= lhs_value.meet(rhs_value);
        }
        return res;
    }
} // namespace potato::trad::analysis
