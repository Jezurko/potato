#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Pass/Pass.h>
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/potato/potato.hpp"

#include "potato/dialect/potato/Passes.hpp"

#include <memory>

namespace potato::pt
{
    // Generate the classes which represent the passes
    #define GEN_PASS_CLASSES
    #include "potato/dialect/potato/Passes.h.inc"

} // namespace potato::pt

