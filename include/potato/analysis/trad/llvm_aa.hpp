#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/lattice.hpp"
#include "potato/analysis/utils.hpp"

// This is a simple Andersen-style alias analysis implementation for llvm dialect
// It's main purpose is to be used for quick testing (or comparsion)
namespace potato::analysis::trad {

struct llaa_lattice : mlir::dataflow::AbstractDenseLattice {

    pt_map< pt_element, lattice_set > pt_relation;

    change_result join(const mlir::dataflow::AbstractDenseLattice &rhs) override;
    change_result meet(const mlir::dataflow::AbstractDenseLattice &rhs) override;
    void print(llvm::raw_ostream &os) const override;
};

struct llvm_andersen : mlir::dataflow::DenseForwardDataFlowAnalysis< llaa_lattice >{
    using base = mlir::dataflow::DenseForwardDataFlowAnalysis< llaa_lattice >;
    using base::propagateIfChanged;

    void visitOperation(mlir::Operation *op, const llaa_lattice &before, llaa_lattice *after) override;
    void visitCallControlFlowTransfer(mlir::CallOpInterface call,
                                      mlir::dataflow::CallControlFlowAction action,
                                      const llaa_lattice &before,
                                      llaa_lattice *after) override;
    void setToEntryState(llaa_lattice *lattice) override;
};
} // namespace potato::trad::analysis
