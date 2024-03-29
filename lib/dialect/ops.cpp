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

mlir::OpFoldResult CopyOp::fold(FoldAdaptor) {
    mlir::OpFoldResult res{};
    for (auto operand : getOperands()) {
        if (!(operand.getDefiningOp()->hasTrait< mlir::OpTrait::ConstantLike >())) {
            // Copy op is joining results of multiple non-constant operations,
            // conservatively bail out to not lose any information
            if (res)
                return {};
            res = operand;
        }
    }
    if (auto operand = mlir::dyn_cast< mlir::Value >(res)) {
        operand.setLoc(mlir::FusedLoc::get(getContext(), {operand.getLoc(), this->getLoc()}));
    }
    return res;
}
