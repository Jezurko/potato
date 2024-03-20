#pragma once

#include <potato/util/warnings.hpp>

POTATO_RELAX_WARNINGS
#include <mlir/IR/Operation.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
POTATO_UNRELAX_WARNINGS

#include <memory>

namespace potato::pt {

    std::unique_ptr< mlir::Pass > createPointsToPass();

    #define GEN_PASS_REGISTRATION
    #include "potato/dialect/potato/Passes.h.inc"

} // namespace potato::pt
