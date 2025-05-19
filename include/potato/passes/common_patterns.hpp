#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/IR/PatternMatch.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>
POTATO_UNRELAX_WARNINGS

#include "potato/util/common.hpp"
#include "potato/dialect/ops.hpp"

namespace potato::conv::cf
{
    struct branch_pattern : mlir::OpInterfaceConversionPattern< branch_iface > {
        using base = mlir::OpInterfaceConversionPattern< branch_iface >;
        using base::base;

        logical_result matchAndRewrite(branch_iface branch,
                                       mlir::ArrayRef< mlir_value > operands,
                                       mlir::ConversionPatternRewriter& rewriter
        ) const override {
            mlir::SmallVector< mlir::ValueRange > ops;
            for (unsigned i = 0; i < branch->getNumSuccessors(); i++) {
                auto succ_ops = branch.getSuccessorOperands(i);
                if (succ_ops.empty()) {
                    ops.emplace_back(mlir::ValueRange());
                } else {
                    auto start_idx = succ_ops.getOperandIndex(0);
                    ops.emplace_back(operands.slice(start_idx, succ_ops.size()));
                }
            }
            rewriter.replaceOpWithNewOp< pt::BranchOp >(branch, ops, branch->getSuccessors());
            return mlir::success();
        }
    };

} // namespace potato::conv::cf
