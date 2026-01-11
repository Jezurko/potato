#pragma once

namespace potato {
#define GEN_PASS_DECL_FUNCTIONMODELLING
#include "potato/passes/conversion/Conversions.h.inc"

std::unique_ptr< mlir::Pass > createFunctionModellingPass();
} // namespace potato
