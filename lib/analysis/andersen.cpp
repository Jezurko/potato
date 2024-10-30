#include "potato/analysis/andersen.hpp"
namespace potato::analysis {

unsigned int aa_lattice::mem_loc_count = 0;

unsigned int aa_lattice::alloc_count() {
    return mem_loc_count++;
}

void aa_lattice::print(llvm::raw_ostream &os) const {
    auto sep = "";
    for (const auto &[key, vals] : *pt_relation) {
        os << sep << key << " -> " << vals;
        sep = "\n";
    }
    os << "\n";
}

llvm::StringRef aa_lattice::get_alloc_name() {
    if (!alloc_name)
        alloc_name = "mem_alloc" + std::to_string(alloc_count());
    return alloc_name.value();
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const potato::analysis::aa_lattice &l) { l.print(os); return os; }
} // namespace potato::analysis

