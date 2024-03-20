#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/potato/potato.hpp"

#include <memory>

namespace potato
{
    std::unique_ptr< mlir::Pass > createLLVMToPotatoPass();

    #define GEN_PASS_REGISTRATION
    #include "potato/conversion/Conversions.h.inc"

    #define GEN_PASS_CLASSES
    #include "potato/conversion/Conversions.h.inc"

} // namespace potato
