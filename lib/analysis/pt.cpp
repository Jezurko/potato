#include "potato/analysis/pt.hpp"

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
    op->walk([&](mlir_operation *op) {
        if (mlir::isa< mlir::ModuleOp >(op))
            return;
        os << "State in: " << op->getLoc() << "\n";
        if (auto state = solver.lookupState< aa_lattice >(op)) {
            for (const auto &[key, vals] : state->pt_relation) {
                os << "  " << key << " -> {";
                if (vals.is_top()) {
                    os << " TOP }\n";
                    continue;
                }
                std::string sep;
                for (const auto &val : vals.get_set_ref()) {
                        os << sep << val;
                        sep = ", ";
                }
                os << "}\n";
            }
        }
    });
}
} // namespace potato::analysis
