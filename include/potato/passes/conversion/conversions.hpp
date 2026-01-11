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
#include "potato/passes/conversion/llvmtopotato.hpp"
#include "potato/passes/conversion/modelling.hpp"

#include <memory>

namespace potato
{
    #define GEN_PASS_REGISTRATION
    #include "potato/passes/conversion/Conversions.h.inc"
} // namespace potato
