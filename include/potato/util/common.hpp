#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataFlow/DenseAnalysis.h>
#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/IR/Value.h>
#include <llvm/ADT/SetVector.h>
POTATO_UNRELAX_WARNINGS

#include <map>
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

using alias_res = mlir::AliasResult;
using alias_kind = mlir::AliasResult::Kind;

using optional_value = std::optional< mlir_value >;
using optional_operation = std::optional< mlir_operation >;
using optional_loc = std::optional< mlir_loc >;

template < typename lattice >
using mlir_dense_dfa = mlir::dataflow::DenseForwardDataFlowAnalysis< lattice >;
using mlir_dense_abstract_lattice = mlir::dataflow::AbstractDenseLattice;
