#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/potato.hpp"

#include <memory>

namespace potato
{
    std::unique_ptr< mlir::Pass > createLLVMToPotatoPass();
    std::unique_ptr< mlir::Pass > createFunctionModellingPass();

    #define GEN_PASS_REGISTRATION
    #include "potato/passes/Conversions.h.inc"

    #define GEN_PASS_CLASSES
    #include "potato/passes/Conversions.h.inc"

} // namespace potato
