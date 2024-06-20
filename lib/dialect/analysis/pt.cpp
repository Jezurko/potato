#include "potato/dialect/analysis/pt.hpp"

namespace potato::analysis {

unsigned int aa_lattice::mem_loc_count = 0;
unsigned int aa_lattice::constant_count = 0;

unsigned int aa_lattice::alloc_count() {
    return mem_loc_count++;
}

unsigned int aa_lattice::const_count() {
    return constant_count++;
}

void aa_lattice::print(llvm::raw_ostream &os) const
{
    for (const auto &[key, vals] : pt_relation) {
        os << key << " -> {";
        if (vals.is_top()) {
            os << " TOP }";
            return;
        }
        std::string sep;
        for (const auto &val : vals.get_set_ref()) {
                os << sep << val;
                sep = ", ";
        }
        os << "}";
    }
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
                    os << " TOP }";
                    return;
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
