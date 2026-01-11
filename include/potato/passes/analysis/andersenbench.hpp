#pragma once

namespace potato::pt {
#define GEN_PASS_DECL_AABENCHPASS
#include "potato/passes/analysis/Analysis.h.inc"

std::unique_ptr< mlir::Pass > createAABenchPass();
} // namespace potato::pt
