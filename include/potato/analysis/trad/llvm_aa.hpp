#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/DenseSet.h>
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/lattice.hpp"
#include "potato/analysis/utils.hpp"

// This is a simple Andersen-style alias analysis implementation for llvm dialect
// It's main purpose is to be used for quick testing (or comparsion)

namespace potato::analysis::trad {

struct llaa_lattice : mlir::dataflow::AbstractDenseLattice {

    using mlir::dataflow::AbstractDenseLattice::AbstractDenseLattice;

    using relation_t = pt_map< pt_element, lattice_set >;
    using set_t = lattice_set< pt_element >;
    relation_t pt_relation;

    static unsigned int variable_count;
    static unsigned int mem_loc_count;
    unsigned int var_count();
    unsigned int alloc_count();

    std::optional< std::string > var_name = {};
    std::optional< std::string > alloc_name = {};
    std::string get_var_name();
    std::string get_alloc_name();

    change_result join(const mlir::dataflow::AbstractDenseLattice &rhs) override;
    change_result meet(const mlir::dataflow::AbstractDenseLattice &rhs) override;
    std::pair< relation_t::iterator, bool > new_var(mlir_value var);
    std::pair< relation_t::iterator, bool > new_var(mlir_value, const set_t &pt_set);
    std::pair< relation_t::iterator, bool > new_var(mlir_value var, mlir_value pointee);
    change_result set_var(mlir_value val, const set_t &pt_set);
    change_result set_var(mlir_value val, mlir_value pointee);
    void print(llvm::raw_ostream &os) const override;
};

namespace {
    namespace mllvm = mlir::LLVM;
}

struct llvm_andersen : mlir::dataflow::DenseForwardDataFlowAnalysis< llaa_lattice >{
    using base = mlir::dataflow::DenseForwardDataFlowAnalysis< llaa_lattice >;
    using base::base;

    using base::propagateIfChanged;

    void visit_alloc(mllvm::AllocaOp &op, const llaa_lattice &before, llaa_lattice *after);
    void visit_store(mllvm::StoreOp &op, const llaa_lattice &before, llaa_lattice *after);
    void visit_load(mllvm::LoadOp &op, const llaa_lattice &before, llaa_lattice *after);
    void visit_constant(mllvm::ConstantOp &op, const llaa_lattice &before, llaa_lattice *after);
    void visit_addresof(mllvm::AddressOfOp &op, const llaa_lattice &before, llaa_lattice *after);
    void visit_cmp(mlir::Operation *op, const llaa_lattice &before, llaa_lattice *after);
    void visit_arith(mlir::Operation *op, const llaa_lattice &before, llaa_lattice *after);

    void visitOperation(mlir::Operation *op, const llaa_lattice &before, llaa_lattice *after) override;
    void visitCallControlFlowTransfer(mlir::CallOpInterface call,
                                      mlir::dataflow::CallControlFlowAction action,
                                      const llaa_lattice &before,
                                      llaa_lattice *after) override;
    void setToEntryState(llaa_lattice *lattice) override;
};

void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os);
} // namespace potato::trad::analysis
