#pragma once

namespace potato::pt {
#define GEN_PASS_DECL_LLVMPOINTSTOPASS
#include "potato/passes/analysis/Analysis.h.inc"

std::unique_ptr< mlir::Pass > createLLVMPointsToPass();
} // namespace potato::pt
