#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/IR/Value.h>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SetVector.h>
POTATO_UNRELAX_WARNINGS

using mlir_value = mlir::Value;
using change_result = mlir::ChangeResult;

using pt_map = llvm::DenseMap< mlir_value, llvm::SetVector< mlir_value > >;

template <typename lattice >
using mlir_dense_dfa = mlir::dataflow::DenseDataFlowAnalysis< lattice >;
using mlir_dense_abstract_lattice = mlir::dataflow::AbstractDenseLattice;
