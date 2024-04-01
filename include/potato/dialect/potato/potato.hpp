#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OperationSupport.h>
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/potato/PotatoDialect.h.inc"
#include "potato/dialect/potato/Potato.h.inc"
#include "potato/dialect/potato/ops.hpp"
#include "potato/dialect/potato/types.hpp"
