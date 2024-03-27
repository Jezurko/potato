#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Support/LLVM.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <llvm/ADT/APSInt.h>
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/potato/potato.hpp"
#include "potato/dialect/potato/ops.hpp"

// TableGen generated stuff goes here:

#include "potato/dialect/potato/PotatoDialect.cpp.inc"

using namespace potato::pt;

#define GET_OP_CLASSES
#include "potato/dialect/potato/Potato.cpp.inc"

mlir::OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
    return mlir::UnitAttr::get(this->getContext());
}

mlir::OpFoldResult ValuedConstantOp::fold(FoldAdaptor adaptor) {
    return adaptor.getValue();
}
