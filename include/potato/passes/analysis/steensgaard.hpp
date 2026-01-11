#pragma once

namespace potato::pt {
#define GEN_PASS_DECL_STEENSGAARDPOINTSTOPASS
#include "potato/passes/analysis/Analysis.h.inc"

std::unique_ptr< mlir::Pass > createSteensgaardPointsToPass();
} // namespace potato::pt
