#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Analysis/CallGraph.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/SymbolTable.h>
#include <mlir/IR/Value.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include <llvm/ADT/SetVector.h>
POTATO_UNRELAX_WARNINGS

#include <optional>

using mlir_type = mlir::Type;
using mlir_value = mlir::Value;
using mlir_operation = mlir::Operation;
using mlir_block = mlir::Block;
using mlir_region = mlir::Region;
using change_result = mlir::ChangeResult;
using mlir_loc = mlir::Location;
using operand_range = mlir::OperandRange;
using result_range = mlir::ResultRange;

using logical_result = mlir::LogicalResult;

using ppoint = mlir::ProgramPoint;
using call_cf_action = mlir::dataflow::CallControlFlowAction;

using symbol_table = mlir::SymbolTable;

using alias_res = mlir::AliasResult;
using alias_kind = mlir::AliasResult::Kind;

using optional_value = std::optional< mlir_value >;
using optional_operation = std::optional< mlir_operation >;
using optional_loc = std::optional< mlir_loc >;

template < typename lattice >
using mlir_dense_dfa = mlir::dataflow::DenseForwardDataFlowAnalysis< lattice >;
using mlir_dense_abstract_lattice = mlir::dataflow::AbstractDenseLattice;

using call_graph = mlir::CallGraph;
using cg_node    = mlir::CallGraphNode;
using cg_edge    = mlir::CallGraphNode::Edge;

using string_ref = llvm::StringRef;

namespace potato::util {
    // Copy-pasted from llvm SetOperations.h with the addition of return value
    // that returns bool whether S1 changed, providing more uniform API for set operations
    template <class S1Ty, class S2Ty>
    bool set_intersect(S1Ty &S1, const S2Ty &S2) {
       bool changed = false;
       for (typename S1Ty::iterator I = S1.begin(); I != S1.end();) {
         const auto &E = *I;
         ++I;
         if (!S2.count(E)) {
            S1.erase(E);   // Erase element if not in S2
            changed = true;
         }
       }
       return changed;
    }

    template< typename analysis_lattice >
    analysis_lattice *get_analysis(mlir::DataFlowSolver &solver, mlir_operation *root) {
        analysis_lattice *lattice = nullptr;
        root->walk([&](mlir_operation *op) -> mlir::WalkResult {
            if(solver.lookupState< analysis_lattice >(op)) {
                // get non-const state for analyses like steensgaard
                // that might modify the state on lookup
                lattice = solver.getOrCreateState< analysis_lattice >(op);
                return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
        });
        return lattice;
    }

    template< typename analysis_lattice >
    void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os)
    {
        auto lattice = get_analysis< analysis_lattice >(solver, op);
        lattice->print(os);
    }

} // namespace potato::util
