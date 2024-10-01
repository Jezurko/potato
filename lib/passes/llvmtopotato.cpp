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
#include <llvm/ADT/TypeSwitch.h>
POTATO_UNRELAX_WARNINGS

#include "potato/passes/conversions.hpp"
#include "potato/passes/type/converter.hpp"
#include "potato/dialect/potato.hpp"
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
            auto tc = this->getTypeConverter();
            rewriter.replaceOpWithNewOp< pt::AllocOp >(op, tc->convertType(op.getRes().getType()));
            return mlir::success();
        }
    };

    using alloc_patterns = util::type_list<
        alloc_op< mlir::LLVM::AllocaOp >
    >;

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

    using store_patterns = util::type_list<
        store_op
    >;

    struct load_op : mlir::OpConversionPattern< mlir::LLVM::LoadOp > {
        using base = mlir::OpConversionPattern< mlir::LLVM::LoadOp >;
        using base::base;
        using adaptor_t = typename mlir::LLVM::LoadOp::Adaptor;

        logical_result matchAndRewrite(mlir::LLVM::LoadOp op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            auto tc = this->getTypeConverter();
            rewriter.replaceOpWithNewOp< pt::DereferenceOp >(
                    op,
                    tc->convertType(op.getRes().getType()),
                    adaptor.getAddr()
            );
            return mlir::success();
        }
    };

    using load_patterns = util::type_list<
        load_op
    >;

    template< typename source >
    struct copy_op : mlir::OpConversionPattern< source > {
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;
        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            auto tc = this->getTypeConverter();
            rewriter.replaceOpWithNewOp< pt::CopyOp >(
                    op,
                    tc->convertType(op.getType()),
                    adaptor.getOperands()
            );
            return mlir::success();
        }
    };

    struct gep_insensitive : mlir::OpConversionPattern< mlir::LLVM::GEPOp > {
        using source = mlir::LLVM::GEPOp;
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;
        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            auto tc = this->getTypeConverter();
            if (op->hasAttr(op.getInboundsAttrName())) {
                rewriter.replaceOpWithNewOp< pt::CopyOp >(
                        op,
                        tc->convertType(op.getType()),
                        adaptor.getBase()
                );
                return mlir::success();
            }
            rewriter.replaceOpWithNewOp< pt::UnknownPtrOp >(
                    op,
                    tc->convertType(op.getType()),
                    adaptor.getOperands()
            );
            return mlir::success();
        }
    };

    struct memcpy_insensitive : mlir::OpConversionPattern< mlir::LLVM::MemcpyOp > {
        using source = mlir::LLVM::MemcpyOp;
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;
        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            // We model the assign op in a way that it rewrites the pt set
            // In this case we need to first merge the pt sets, because the memcpy
            // may rewrite only part of the target location.
            auto merged = rewriter.create< pt::CopyOp >(op.getLoc(),
                                          pt::PointerType::get(rewriter.getContext()),
                                          mlir::ValueRange{adaptor.getSrc(), adaptor.getDst()}
            );

            rewriter.replaceOpWithNewOp< pt::AssignOp >(
                    op,
                    adaptor.getDst(),
                    merged
            );
            return mlir::success();
        }
    };


    using copy_patterns = util::type_list<
        copy_op< mlir::LLVM::AddOp >,
        copy_op< mlir::LLVM::FAddOp >,
        copy_op< mlir::LLVM::FMulAddOp >,
        copy_op< mlir::LLVM::FNegOp >,
        copy_op< mlir::LLVM::SubOp >,
        copy_op< mlir::LLVM::FSubOp >,
        copy_op< mlir::LLVM::MulOp >,
        copy_op< mlir::LLVM::FMulOp >,
        copy_op< mlir::LLVM::MulOp >,
        copy_op< mlir::LLVM::FMulOp >,
        copy_op< mlir::LLVM::SDivOp >,
        copy_op< mlir::LLVM::UDivOp >,
        copy_op< mlir::LLVM::FDivOp >,
        copy_op< mlir::LLVM::TruncOp >,
        copy_op< mlir::LLVM::PtrToIntOp >,
        copy_op< mlir::LLVM::IntToPtrOp >,
        copy_op< mlir::LLVM::BitcastOp >,
        copy_op< mlir::LLVM::ZExtOp >,
        copy_op< mlir::LLVM::SExtOp >,
        gep_insensitive,
        memcpy_insensitive
    >;

    template< typename source >
    struct constant_op : mlir::OpConversionPattern< source > {
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;
        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            auto tc = this->getTypeConverter();
            rewriter.replaceOpWithNewOp< pt::ConstantOp >(op, tc->convertType(op.getType()));
            return mlir::success();
        }
    };

    struct val_constant_op : mlir::OpConversionPattern< mlir::LLVM::ConstantOp > {
        using base = mlir::OpConversionPattern< mlir::LLVM::ConstantOp >;
        using base::base;
        using adaptor_t = typename mlir::LLVM::ConstantOp::Adaptor;

        logical_result matchAndRewrite(mlir::LLVM::ConstantOp op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            auto const_attr = op.getValue();
            auto builder = [&](auto attr) {
                rewriter.replaceOpWithNewOp< pt::ValuedConstantOp >(op, op.getType(), attr);
                return mlir::success();
            };

            return llvm::TypeSwitch< mlir::Attribute, logical_result >(const_attr)
                .Case< mlir::BoolAttr,
                       mlir::FloatAttr,
                       mlir::IntegerAttr
                 >(builder)
                .Default([&](auto) {return mlir::failure();});
        }
    };

    using constant_patterns = util::type_list<
        constant_op< mlir::LLVM::ConstantOp >,
        constant_op< mlir::LLVM::ZeroOp >,
        constant_op< mlir::LLVM::ICmpOp >,
        constant_op< mlir::LLVM::FCmpOp >
    >;

    struct address_of_op : mlir::OpConversionPattern< mlir::LLVM::AddressOfOp > {
        using base = mlir::OpConversionPattern< mlir::LLVM::AddressOfOp >;
        using base::base;
        using adaptor_t = typename mlir::LLVM::AddressOfOp::Adaptor;

        logical_result matchAndRewrite(mlir::LLVM::AddressOfOp op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            rewriter.replaceOpWithNewOp< pt::AddressOp >(
                    op,
                    this->getTypeConverter()->convertType(op.getRes().getType()),
                    mlir::Value(),
                    op.getGlobalNameAttr()
            );
            return mlir::success();
        }
    };

    using address_patterns = util::type_list<
        address_of_op
    >;

    struct potato_target : public mlir::ConversionTarget {
        potato_target(mlir::MLIRContext &ctx) : ConversionTarget(ctx) {
            addLegalDialect< pt::PotatoDialect >();
        }
    };

    using pattern_list = util::concat<
        alloc_patterns,
        constant_patterns,
        copy_patterns,
        store_patterns,
        load_patterns,
        address_patterns
    >;

    struct LLVMIRToPoTAToPass : LLVMIRToPoTAToBase< LLVMIRToPoTAToPass >
    {
        template< typename list >
        void add_patterns(mlir::RewritePatternSet &patterns, auto &converter) {
            if constexpr (list::empty) {
                return;
            } else {
                patterns.add< typename list::head >(converter, patterns.getContext());
                return add_patterns< typename list::tail >(patterns, converter);
            }
        }

        void runOnOperation() override {
            auto &mctx    = getContext();
            auto tc       = to_pt_type();
            auto trg      = potato_target(mctx);
            auto patterns = mlir::RewritePatternSet(&mctx);

            add_patterns< pattern_list >(patterns, tc);

            trg.addDynamicallyLegalDialect< mlir::LLVM::LLVMDialect >(
                    [&](auto *op){
                        return mlir::isa< mlir::BranchOpInterface,
                                          mlir::CallOpInterface,
                                          mlir::FunctionOpInterface,
                                          mlir::RegionBranchOpInterface,
                                          mlir::LLVM::ReturnOp,
                                          mlir::LLVM::GlobalOp
                                        > (op);
            });

            if (failed(applyPartialConversion(getOperation(),
                                       trg,
                                       std::move(patterns))))
                    return signalPassFailure();
        }
    };

} // namespace potato::conv::llvmtopt

std::unique_ptr< mlir::Pass > potato::createLLVMToPotatoPass() {
    return std::make_unique< potato::conv::llvmtopt::LLVMIRToPoTAToPass >();
}

