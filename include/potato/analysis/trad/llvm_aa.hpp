#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/andersen.hpp"

#include <memory>

// This is a simple Andersen-style alias analysis implementation for llvm dialect
// It's main purpose is to be used for quick testing (or comparsion)

namespace potato::analysis::trad {

namespace {
    namespace mllvm = mlir::LLVM;
}

struct llvm_andersen : mlir::dataflow::DenseForwardDataFlowAnalysis< aa_lattice >{
    using base = mlir::dataflow::DenseForwardDataFlowAnalysis< aa_lattice >;
    using base::base;

    using base::propagateIfChanged;

    void visit_op(mllvm::AllocaOp &op, const aa_lattice &before, aa_lattice *after);
    void visit_op(mllvm::StoreOp &op, const aa_lattice &before, aa_lattice *after);
    void visit_op(mllvm::LoadOp &op, const aa_lattice &before, aa_lattice *after);
    void visit_op(mllvm::ConstantOp &op, const aa_lattice &before, aa_lattice *after);
    void visit_op(mllvm::ZeroOp &op, const aa_lattice &before, aa_lattice *after);
    void visit_op(mllvm::GEPOp &op, const aa_lattice &before, aa_lattice *after);
    void visit_op(mllvm::AddressOfOp &op, const aa_lattice &before, aa_lattice *after);
    void visit_op(mllvm::SExtOp &op, const aa_lattice &before, aa_lattice *after);
    void visit_op(mllvm::GlobalOp &op, const aa_lattice &before, aa_lattice *after);
    void visit_op(mllvm::MemcpyOp &op, const aa_lattice &before, aa_lattice *after);
    void visit_op(mllvm::SelectOp &op, const aa_lattice &before, aa_lattice *after);

    void visit_op(mlir::BranchOpInterface &op, const aa_lattice &before, aa_lattice *after);

    void visit_cmp(mlir::Operation *op, const aa_lattice &before, aa_lattice *after);
    void visit_arith(mlir::Operation *op, const aa_lattice &before, aa_lattice *after);
    std::vector< mlir::Operation * > get_function_returns(mlir::FunctionOpInterface func);
    std::vector< const aa_lattice * > get_or_create_for(mlir::Operation * dep, const std::vector< mlir::Operation * > &ops);

    void visitOperation(mlir::Operation *op, const aa_lattice &before, aa_lattice *after) override;
    void visitCallControlFlowTransfer(mlir::CallOpInterface call,
                                      mlir::dataflow::CallControlFlowAction action,
                                      const aa_lattice &before,
                                      aa_lattice *after) override;
    void setToEntryState(aa_lattice *lattice) override;
    mlir::LogicalResult initialize(mlir_operation *op) override;

    llvm_andersen(mlir::DataFlowSolver &solver)
        : base(solver),
          relation(std::make_unique< aa_lattice::relation_t >())
        {}

    private:
    std::unique_ptr< aa_lattice::relation_t > relation;
};

void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);

//void print_analysis_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);
//
//void print_analysis_func_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);
} // namespace potato::trad::analysis
