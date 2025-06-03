#pragma once

#include "potato/dialect/potato.hpp"

#define GET_OP_CLASSES

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
#include "potato/dialect/Potato.h.inc"
POTATO_UNRELAX_WARNINGS
