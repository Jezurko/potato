#pragma once

#include "potato/util/warnings.hpp"


POTATO_RELAX_WARNINGS
#include <mlir/Dialect/SCF/IR/SCF.h>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
POTATO_UNRELAX_WARNINGS

namespace potato
{
    std::unique_ptr< mlir::Pass > createLLVMToPotatoPass();
} // namespace potato
