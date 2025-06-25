#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
POTATO_UNRELAX_WARNINGS

#include "potato/passes/conversions.hpp"
#include "potato/util/common.hpp"

namespace potato::conv::modelling
{
    struct FunctionModellingPass : FunctionModellingBase< FunctionModellingPass >
    {
        void runOnOperation() override {
            if (inline_bodies) {
                llvm::errs() << "Inlining NYI for modelling pass\n";
                return signalPassFailure();
            }
            auto &mctx = getContext();
            mlir::ModuleOp root = getOperation();
        }
    };
} // namespace potato::conv::modelling

std::unique_ptr< mlir::Pass > potato::createFunctionModellingPass() {
    return std::make_unique< potato::conv::modelling::FunctionModellingPass >();
}
