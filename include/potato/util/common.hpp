#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/IR/Value.h>
#include <llvm/ADT/SetVector.h>
POTATO_UNRELAX_WARNINGS

#include <map>
#include <optional>

using mlir_value = mlir::Value;
using mlir_operation = mlir::Operation;
using mlir_block = mlir::Block;
using mlir_region = mlir::Region;
using change_result = mlir::ChangeResult;
using mlir_location = mlir::Location;

using ppoint = mlir::ProgramPoint;

using optional_value = std::optional< mlir_value >;
using optional_operation = std::optional< mlir_operation >;
using optional_location = std::optional< mlir_location>;

template <typename lattice >
using mlir_dense_dfa = mlir::dataflow::DenseDataFlowAnalysis< lattice >;
using mlir_dense_abstract_lattice = mlir::dataflow::AbstractDenseLattice;
