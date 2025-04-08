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
    struct branch_pattern : mlir::OpInterfaceRewritePattern< branch_iface > {
        using base = mlir::OpInterfaceRewritePattern< branch_iface >;
        using base::base;

        logical_result matchAndRewrite(branch_iface branch,
                                       mlir::PatternRewriter& rewriter
        ) const override {
            mlir::SmallVector< mlir::ValueRange > ops;
            for (unsigned i = 0; i < branch->getNumSuccessors(); i++) {
                ops.push_back(branch.getSuccessorOperands(i).getForwardedOperands());
            }
            rewriter.replaceOpWithNewOp< pt::BranchOp >(branch, ops, branch->getSuccessors());
            return mlir::success();
        }
    };

} // namespace potato::conv::cf
