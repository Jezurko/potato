#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>
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

using logical_result = mlir::LogicalResult;

using ppoint = mlir::ProgramPoint;
using call_cf_action = mlir::dataflow::CallControlFlowAction;

using alias_res = mlir::AliasResult;
using alias_kind = mlir::AliasResult::Kind;

using optional_value = std::optional< mlir_value >;
using optional_operation = std::optional< mlir_operation >;
using optional_loc = std::optional< mlir_loc >;

template < typename lattice >
using mlir_dense_dfa = mlir::dataflow::DenseForwardDataFlowAnalysis< lattice >;
using mlir_dense_abstract_lattice = mlir::dataflow::AbstractDenseLattice;

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
    void print_analysis_result(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os)
    {
        op->walk([&](mlir_operation *op) {
            if (mlir::isa< mlir::ModuleOp >(op))
                return;
            os << "State in: " << op->getLoc() << "\n";
            if (auto state = solver.lookupState< analysis_lattice >(op)) {
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

    template< typename analysis_lattice >
    void print_analysis_stats(mlir::DataFlowSolver &solver, mlir_operation *op, llvm::raw_ostream &os)
    {
        int top_glob = 0, bottom_glob = 0, single_elem_glob = 0, multiple_elem_glob = 0;
        op->walk([&](mlir_operation *op) {
            if (mlir::isa< mlir::ModuleOp >(op))
                return;
            int top = 0, bottom = 0, single_elem = 0, multiple_elem = 0;
            if (auto state = solver.lookupState< analysis_lattice >(op)) {
                for (const auto &[key, vals] : state->pt_relation) {
                    if (vals.is_top())
                        top++;

                    if (vals.is_bottom())
                        bottom++;

                    if (vals.is_concrete()) {
                        if (vals.is_single_target())
                            single_elem++;
                        else
                            multiple_elem++;
                    }
                }
            top_glob += top;
            bottom_glob += bottom;
            single_elem_glob += single_elem;
            multiple_elem_glob += multiple_elem;
            os << "State in: " << op->getLoc() << "\n";
            os << "Tops: " << top
               << " Bottoms: " << bottom
               << " Single element: " << single_elem
               << " Multiple element: " << multiple_elem;
            os << "\n";
            }
        });
        os << "Global state\n";
        os << "Tops: " << top_glob
           << " Bottoms: " << bottom_glob
           << " Single element: " << single_elem_glob
           << " Multiple element: " << multiple_elem_glob;
        os << "\n";
    }
} // namespace potato::util
