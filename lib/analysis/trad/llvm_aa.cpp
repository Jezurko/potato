#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
POTATO_UNRELAX_WARNINGS

#include "potato/analysis/trad/llvm_aa.hpp"
#include "potato/util/common.hpp"

namespace potato::analysis::trad {
} // namespace potato::trad::analysis
