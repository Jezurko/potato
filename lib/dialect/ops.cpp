#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/APSInt.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LLVM.h>
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/potato.hpp"
#include "potato/dialect/ops.hpp"

// TableGen generated stuff goes here:

#include "potato/dialect/PotatoDialect.cpp.inc"

using namespace potato::pt;

#define GET_OP_CLASSES
#include "potato/dialect/Potato.cpp.inc"

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
        if (mlir::isa< pt::UnknownPtrOp >(operand.getDefiningOp())) {
            return operand;
        }
    }
    if (!res && this->getNumOperands() > 0) {
        res = getOperand(0);
    }
    if (auto operand = mlir::dyn_cast_if_present< mlir::Value >(res)) {
        operand.setLoc(mlir::FusedLoc::get(getContext(), {operand.getLoc(), this->getLoc()}));
    }
    return res;
}
