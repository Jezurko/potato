#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Analysis/DataLayoutAnalysis.h>
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/SmallVector.h>
POTATO_UNRELAX_WARNINGS

#include "potato/conversion/conversions.hpp"
#include "potato/dialect/potato/potato.hpp"
#include "potato/util/common.hpp"
#include "potato/util/typelist.hpp"

namespace potato::conv::llvmtopt
{

    template< typename source >
    struct alloc_op : mlir::OpConversionPattern< source > {
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;

        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            rewriter.replaceOpWithNewOp< pt::AllocOp >(op, op.getRes().getType());
            return mlir::success();
        }
    };

    struct store_op : mlir::OpConversionPattern< mlir::LLVM::StoreOp > {
        using base = mlir::OpConversionPattern< mlir::LLVM::StoreOp >;
        using base::base;
        using adaptor_t = typename mlir::LLVM::StoreOp::Adaptor;

        logical_result matchAndRewrite(mlir::LLVM::StoreOp op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            rewriter.replaceOpWithNewOp< pt::AssignOp >(op, adaptor.getAddr(), adaptor.getValue());
            return mlir::success();
        }
    };

    struct load_op : mlir::OpConversionPattern< mlir::LLVM::LoadOp > {
        using base = mlir::OpConversionPattern< mlir::LLVM::LoadOp >;
        using base::base;
        using adaptor_t = typename mlir::LLVM::LoadOp::Adaptor;

        logical_result matchAndRewrite(mlir::LLVM::LoadOp op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            rewriter.replaceOpWithNewOp< pt::DereferenceOp >(op, op.getRes().getType(), adaptor.getAddr());
            return mlir::success();
        }
    };

    struct potato_target : public mlir::ConversionTarget {
        potato_target(mlir::MLIRContext &ctx) : ConversionTarget(ctx) {
            addLegalDialect< pt::PotatoDialect >();

            addDynamicallyLegalDialect< mlir::LLVM::LLVMDialect >(
                    [&](auto *op){
                        return mlir::isa< mlir::BranchOpInterface,
                                          mlir::RegionBranchOpInterface,
                                          mlir::CallOpInterface,
                                          mlir::FunctionOpInterface
                                        > (op);
            });
        }
    };

    using pattern_list = util::type_list<
        alloc_op< mlir::LLVM::AllocaOp >,
        alloc_op< mlir::LLVM::AddOp >,
        store_op,
        load_op
    >;

    struct LLVMIRToPoTAToPass : LLVMIRToPoTAToBase< LLVMIRToPoTAToPass >
    {
        template< typename list >
        void add_patterns(mlir::RewritePatternSet &patterns) {
            if constexpr (list::empty) {
                return;
            } else {
                patterns.add< typename list::head >(patterns.getContext());
                return add_patterns< typename list::tail >(patterns);
            }
        }

        void runOnOperation() override {
            mlir::RewritePatternSet patterns(&getContext());
            add_patterns< pattern_list >(patterns);
        }
    };

} // namespace potato::conv::llvmtopt
