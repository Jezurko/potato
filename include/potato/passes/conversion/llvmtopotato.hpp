#pragma once

namespace potato {
#define GEN_PASS_DECL_LLVMIRTOPOTATO
#include "potato/passes/conversion/Conversions.h.inc"

std::unique_ptr< mlir::Pass > createLLVMToPotatoPass();
} // namespace potato
