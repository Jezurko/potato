#include "potato/analysis/andersen.hpp"
namespace potato::analysis {

void aa_lattice::print(llvm::raw_ostream &os) const {
    auto sep = "";
    for (const auto &[key, vals] : *pt_relation) {
        os << sep << key << " -> " << vals;
        sep = "\n";
    }
    os << "\n";
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const potato::analysis::aa_lattice &l) { l.print(os); return os; }
} // namespace potato::analysis

