#include "potato/analysis/fsanalysis.hpp"
namespace potato::analysis {

unsigned int fs_lattice::variable_count = 0;
unsigned int fs_lattice::mem_loc_count = 0;

unsigned int fs_lattice::var_count() {
    return variable_count++;
}

unsigned int fs_lattice::alloc_count() {
    return mem_loc_count++;
}

void fs_lattice::print(llvm::raw_ostream &os) const {
    for (const auto &[key, vals] : pt_relation) {
        os << key << " -> " << vals;
    }
}

std::string fs_lattice::get_var_name() {
    if (!var_name)
        var_name = "var" + std::to_string(var_count());
    return var_name.value();
}

std::string fs_lattice::get_alloc_name() {
    if (!alloc_name)
        alloc_name = "mem_alloc" + std::to_string(alloc_count());
    return alloc_name.value();
}
} // namespace potato::analysis
