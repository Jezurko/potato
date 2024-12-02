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

    struct memcpy_insensitive : mlir::OpConversionPattern< mlir::LLVM::MemcpyOp > {
        using source = mlir::LLVM::MemcpyOp;
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;
        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            auto src_deref = rewriter.create< pt::DereferenceOp >(
                    op.getLoc(),
                    this->getTypeConverter()->convertType(adaptor.getSrc().getType()),
                    adaptor.getSrc()
            );
            rewriter.replaceOpWithNewOp< pt::AssignOp >(
                    op,
                    adaptor.getDst(),
                    src_deref
            );
            return mlir::success();
        }
    };

    struct memset_insensitive : mlir::OpConversionPattern< mlir::LLVM::MemsetOp > {
        using source = mlir::LLVM::MemsetOp;
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;
        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            rewriter.replaceOpWithNewOp< pt::AssignOp >(
                    op,
                    adaptor.getDst(),
                    adaptor.getVal()
            );
            return mlir::success();
        }
    };

    struct va_start : mlir::OpConversionPattern< mlir::LLVM::VaStartOp > {
        using source = mlir::LLVM::VaStartOp ;
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;
        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            auto paren_fn = op->getParentOfType< mlir::FunctionOpInterface >();
            auto new_alloc = rewriter.create< pt::AllocOp >(op.getLoc(), this->getTypeConverter()->convertType(adaptor.getArgList().getType()));
            rewriter.create< pt::AssignOp >(
                op.getLoc(),
                new_alloc,
                paren_fn.getArgument(paren_fn.getNumArguments() - 1)
            );
            rewriter.replaceOpWithNewOp< pt::AssignOp >(
                 op,
                 adaptor.getArgList(),
                 new_alloc
            );
            return mlir::success();
        }
    };


    using store_patterns = util::type_list<
        store_op,
        memcpy_insensitive,
        memset_insensitive,
        va_start
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
            rewriter.replaceOpWithNewOp< pt::CopyOp >(
                        op,
                        tc->convertType(op.getType()),
                        adaptor.getBase()
                );
            return mlir::success();
        }
    };

    struct insert_value : mlir::OpConversionPattern< mlir::LLVM::InsertValueOp > {
        using source = mlir::LLVM::InsertValueOp;
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
                    mlir::ValueRange{adaptor.getContainer(), adaptor.getValue()}
            );
            return mlir::success();
        }
    };

    struct select_insensitive : mlir::OpConversionPattern< mlir::LLVM::SelectOp > {
        using source = mlir::LLVM::SelectOp;
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;
        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            // TODO: Check if we know the value of cond to select an operand
            rewriter.replaceOpWithNewOp< pt::CopyOp >(
                    op,
                    this->getTypeConverter()->convertType(op.getType()),
                    mlir::ValueRange{adaptor.getTrueValue(), adaptor.getFalseValue()}
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
        copy_op< mlir::LLVM::FPToSIOp >,
        copy_op< mlir::LLVM::FPToUIOp >,
        copy_op< mlir::LLVM::UIToFPOp >,
        copy_op< mlir::LLVM::SIToFPOp >,
        copy_op< mlir::LLVM::SDivOp >,
        copy_op< mlir::LLVM::UDivOp >,
        copy_op< mlir::LLVM::FDivOp >,
        copy_op< mlir::LLVM::TruncOp >,
        copy_op< mlir::LLVM::PtrToIntOp >,
        copy_op< mlir::LLVM::BitcastOp >,
        copy_op< mlir::LLVM::ZExtOp >,
        copy_op< mlir::LLVM::SExtOp >,
        copy_op< mlir::LLVM::FAbsOp >,
        copy_op< mlir::LLVM::URemOp >,
        copy_op< mlir::LLVM::SRemOp >,
        copy_op< mlir::LLVM::FRemOp >,
        gep_insensitive,
        insert_value,
        select_insensitive
    >;

    template< typename source >
    struct unknown_op : mlir::OpConversionPattern< source > {
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;
        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            auto tc = this->getTypeConverter();
            rewriter.replaceOpWithNewOp< pt::UnknownPtrOp >(op, tc->convertType(op.getType()), mlir::ValueRange());
            return mlir::success();
        }
    };

    using unknown_patterns = util::type_list<
        unknown_op< mlir::LLVM::IntToPtrOp >
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
            if (mlir::isa< mlir::LLVM::LLVMPointerType >(op.getType())) {
                rewriter.replaceOpWithNewOp< pt::UnknownPtrOp >(op, tc->convertType(op.getType()), mlir::ValueRange());
            } else {
                rewriter.replaceOpWithNewOp< pt::ConstantOp >(op, tc->convertType(op.getType()));
            }
            return mlir::success();
        }
    };

    struct zero_op : mlir::OpConversionPattern< mlir::LLVM::ZeroOp > {
        using base = mlir::OpConversionPattern< mlir::LLVM::ZeroOp >;
        using base::base;
        using adaptor_t = typename mlir::LLVM::ZeroOp::Adaptor;
        logical_result matchAndRewrite(mlir::LLVM::ZeroOp op,
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
        constant_op< mlir::LLVM::UndefOp >,
        constant_op< mlir::LLVM::ICmpOp >,
        constant_op< mlir::LLVM::FCmpOp >,
        zero_op
    >;

    struct global_op : mlir::OpConversionPattern< mlir::LLVM::GlobalOp > {
        using base = mlir::OpConversionPattern< mlir::LLVM::GlobalOp >;
        using base::base;
        using adaptor_t = mlir::LLVM::GlobalOp::Adaptor;

        logical_result matchAndRewrite(
            mlir::LLVM::GlobalOp op,
            adaptor_t adaptor,
            mlir::ConversionPatternRewriter &rewriter
        ) const override {
            auto global = rewriter.replaceOpWithNewOp< pt::GlobalVarOp >(op, op.getName(), false);
            auto &orig_init = adaptor.getInitializer();
            auto &glob_init = global.getInit();
            if (!orig_init.empty()) {
                rewriter.inlineRegionBefore(orig_init, glob_init, glob_init.end());
                return mlir::success();
            }
            if (auto val_attr = op.getValue()) {
               auto guard = mlir::OpBuilder::InsertionGuard(rewriter);
               rewriter.setInsertionPointToStart(&glob_init.emplaceBlock());
               // TODO: init pointers to unknown
               auto constant = rewriter.create< pt::ConstantOp >(
                    op.getLoc(),
                    this->getTypeConverter()->convertType(op.getGlobalType())
               );
               auto cast = rewriter.create< mlir::UnrealizedConversionCastOp >(
                   op.getLoc(),
                   op.getGlobalType(),
                   constant.getResult()
               );
               rewriter.create< mlir::LLVM::ReturnOp >(op.getLoc(), cast.getOutputs());
            }
            return mlir::success();
        }

    };

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

    struct fix_addr_type_pattern : mlir::OpConversionPattern< pt::AddressOp > {
        using base = mlir::OpConversionPattern< pt::AddressOp >;
        using base::base;
        using adaptor_t = typename pt::AddressOp::Adaptor;

        logical_result matchAndRewrite(pt::AddressOp op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            rewriter.replaceOpWithNewOp< pt::AddressOp >(
                    op,
                    this->getTypeConverter()->convertType(op.getResult().getType()),
                    mlir::Value(),
                    op.getSymbolAttr()
            );
            return mlir::success();
        }
    };

    using global_handling_patterns = util::type_list<
        fix_addr_type_pattern
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
        unknown_patterns,
        global_handling_patterns
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
            auto dummy_tc = mlir::LLVMTypeConverter(&mctx);

            auto address_of_trg      = potato_target(mctx);
            auto address_of_patterns = mlir::RewritePatternSet(&mctx);
            add_patterns< util::type_list< address_of_op > >(address_of_patterns, dummy_tc);

            address_of_trg.addDynamicallyLegalDialect< mlir::LLVM::LLVMDialect >(
                    [&](auto *op){
                        return !mlir::isa< mlir::LLVM::AddressOfOp > (op);
            });

            auto global_trg      = potato_target(mctx);
            auto global_patterns = mlir::RewritePatternSet(&mctx);
            add_patterns< util::type_list< global_op > >(global_patterns, dummy_tc);

            global_trg.addDynamicallyLegalDialect< mlir::LLVM::LLVMDialect >(
                    [&](auto *op){
                        return !mlir::isa< mlir::LLVM::GlobalOp > (op);
            });

            auto trg      = potato_target(mctx);
            auto patterns = mlir::RewritePatternSet(&mctx);

            add_patterns< pattern_list >(patterns, tc);

            trg.addDynamicallyLegalDialect< mlir::LLVM::LLVMDialect >(
                    [&](auto *op){
                        return mlir::isa< mlir::BranchOpInterface,
                                          mlir::FunctionOpInterface,
                                          mlir::RegionBranchOpInterface,
                                          mlir::CallOpInterface,
                                          mlir::LLVM::ReturnOp,
                                          mlir::LLVM::NoAliasScopeDeclOp,
                                          mlir::LLVM::UnreachableOp,
                                          mlir::LLVM::AssumeOp,
                                          mlir::LLVM::VaEndOp
                                        > (op);
            });

            trg.addDynamicallyLegalOp< pt::AddressOp >([&](pt::AddressOp op) {
                return tc.isLegal(op);
            });
            trg.addLegalOp< mlir::UnrealizedConversionCastOp >();

            if (failed(applyPartialConversion(getOperation(),
                                       address_of_trg,
                                       std::move(address_of_patterns))))
                    return signalPassFailure();

            if (failed(applyPartialConversion(getOperation(),
                                       global_trg,
                                       std::move(global_patterns))))
                    return signalPassFailure();

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

