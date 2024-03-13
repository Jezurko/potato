#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/IR/Value.h>
#include <llvm/ADT/SetVector.h>
POTATO_UNRELAX_WARNINGS

#include <map>
#include <optional>

using value = mlir::Value;
using operation = mlir::Operation;
using change_result = mlir::ChangeResult;
using location = mlir::Location;

using optional_value = std::optional< value >;
using optional_operation = std::optional< operation >;
using optional_location = std::optional< location>;

template <typename lattice >
using mlir_dense_dfa = mlir::dataflow::DenseDataFlowAnalysis< lattice >;
using mlir_dense_abstract_lattice = mlir::dataflow::AbstractDenseLattice;
