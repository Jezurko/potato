#include "potato/dialect/potato/analysis/pt.hpp"

namespace potato::analysis {

unsigned int pt_lattice::mem_loc_count = 0;

unsigned int pt_lattice::alloc_count() {
    return mem_loc_count++;
}

void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os)
{
    op->walk([&](mlir_operation *op) {
        if (mlir::isa< mlir::ModuleOp >(op))
            return;
        os << "State in: " << op->getLoc() << "\n";
        if (auto state = solver.lookupState< pt_lattice >(op)) {
            for (const auto &[key, vals] : state->pt_relation) {
                os << "  " << key << " -> {";
                std::string sep;
                for (const auto &val : vals) {
                        os << sep << val;
                        sep = ", ";
                }
                os << "}\n";
            }
        }
    });
}
} // namespace potato::analysis
