#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>

#include "potato/dialect/PotatoDialect.h.inc"
#include "potato/dialect/Potato.h.inc"
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/ops.hpp"
#include "potato/dialect/types.hpp"
