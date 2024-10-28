#include "potato/analysis/pt.hpp"
#include "potato/analysis/fsandersen.hpp"
#include "potato/analysis/context.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis {
void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os)
{
    potato::util::print_analysis_result< call_context_wrapper< fsa_lattice > >(solver, op, os);
}
} // namespace potato::analysis
