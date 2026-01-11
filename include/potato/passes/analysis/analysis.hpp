#pragma once

#include <potato/util/warnings.hpp>

POTATO_RELAX_WARNINGS
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
POTATO_UNRELAX_WARNINGS

#include <potato/passes/analysis/andersen.hpp>
#include <potato/passes/analysis/andersenbench.hpp>
#include <potato/passes/analysis/steensgaard.hpp>
#include <potato/passes/analysis/steensgaardbench.hpp>
#include <potato/passes/analysis/llvmandersen.hpp>

namespace potato::pt {
    #define GEN_PASS_REGISTRATION
    #include "potato/passes/analysis/Analysis.h.inc"
} // namespace potato::pt
