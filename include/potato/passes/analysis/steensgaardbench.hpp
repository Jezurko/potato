#pragma once

namespace potato::pt {
#define GEN_PASS_DECL_STEENSBENCHPASS
#include "potato/passes/analysis/Analysis.h.inc"

std::unique_ptr< mlir::Pass > createSteensBenchPass();
} // namespace potato::pt
