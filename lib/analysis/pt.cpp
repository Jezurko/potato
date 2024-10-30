#include "potato/analysis/pt.hpp"
#include "potato/analysis/andersen.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis {
void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os)
{
    potato::util::print_analysis_result< aa_lattice >(solver, op, os);
}
} // namespace potato::analysis
