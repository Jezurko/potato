#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/andersen.hpp"
#include "potato/analysis/pt.hpp"

// This is a simple Andersen-style alias analysis implementation for llvm dialect
// It's main purpose is to be used for quick testing (or comparsion)

namespace potato::analysis::trad {

namespace {
    namespace mllvm = mlir::LLVM;
}

struct llvm_andersen : pt_analysis< llvm_andersen, aa_lattice > {
    using base = pt_analysis< llvm_andersen, aa_lattice >;
    using base::base;

    void register_anchors() {}
    void set_to_entry_state(aa_lattice *lattice) {}

    logical_result visit_operation(
        mlir_operation *op, const_lattices_ref operand_lts, lattices_ref res_lts
    );
};

void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);

} // namespace potato::trad::analysis
