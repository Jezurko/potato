#pragma once

namespace potato::pt {
#define GEN_PASS_DECL_POINTSTOPASS
#include "potato/passes/analysis/Analysis.h.inc"

std::unique_ptr< mlir::Pass > createPointsToPass();
} // namespace potato::pt
