#include "potato/analysis/pt.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis {

unsigned int aa_lattice::variable_count = 0;
unsigned int aa_lattice::mem_loc_count = 0;

unsigned int aa_lattice::var_count() {
    return variable_count++;
}

unsigned int aa_lattice::alloc_count() {
    return mem_loc_count++;
}

void aa_lattice::print(llvm::raw_ostream &os) const {
    for (const auto &[key, vals] : pt_relation) {
        os << key << " -> " << vals;
    }
}

std::string aa_lattice::get_var_name() {
    if (!var_name)
        var_name = "var" + std::to_string(var_count());
    return var_name.value();
}

std::string aa_lattice::get_alloc_name() {
    if (!alloc_name)
        alloc_name = "mem_alloc" + std::to_string(alloc_count());
    return alloc_name.value();
}

void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os)
{
    potato::util::print_analysis_result< aa_lattice >(solver, op, os);
}

void print_analysis_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os)
{
    potato::util::print_analysis_stats< aa_lattice >(solver, op, os);
}

void print_analysis_func_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os)
{
    potato::util::print_analysis_func_stats< aa_lattice >(solver, op, os);
}
} // namespace potato::analysis
