#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/Conversion/LLVMCommon/Pattern.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/TypeSwitch.h>
POTATO_UNRELAX_WARNINGS

#include "potato/passes/conversion/conversions.hpp"
#include "potato/passes/conversion/common_patterns.hpp"
#include "potato/passes/conversion/type/converter.hpp"
#include "potato/dialect/potato.hpp"
#include "potato/util/common.hpp"
#include "potato/util/typelist.hpp"

namespace potato::conv::llvmtopt {
#define GEN_PASS_DEF_LLVMIRTOPOTATO
#include "potato/passes/conversion/Conversions.h.inc"

    template< typename source >
    struct alloc_op : mlir::OpConversionPattern< source > {
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;

        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            rewriter.replaceOpWithNewOp< pt::AllocOp >(
                    op,
                    this->typeConverter->convertType(op.getRes().getType())
            );
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

    struct cmpxchg_op : mlir::OpConversionPattern< mlir::LLVM::AtomicCmpXchgOp > {
        using base = mlir::OpConversionPattern< mlir::LLVM::AtomicCmpXchgOp >;
        using base::base;
        using adaptor_t = typename mlir::LLVM::AtomicCmpXchgOp::Adaptor;

        logical_result matchAndRewrite(mlir::LLVM::AtomicCmpXchgOp op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            rewriter.replaceOpWithNewOp< pt::DereferenceOp >(
                    op,
                    typeConverter->convertType(op.getRes().getType()),
                    adaptor.getPtr()
            );
            pt::AssignOp::create(rewriter, op.getLoc(), adaptor.getPtr(), adaptor.getVal());
            return mlir::success();
        }
    };

    struct atomic_rmw : mlir::OpConversionPattern< mlir::LLVM::AtomicRMWOp > {
        using base = mlir::OpConversionPattern< mlir::LLVM::AtomicRMWOp >;
        using base::base;
        using adaptor_t = typename mlir::LLVM::AtomicRMWOp::Adaptor;

        logical_result matchAndRewrite(mlir::LLVM::AtomicRMWOp op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            rewriter.replaceOpWithNewOp< pt::DereferenceOp >(
                    op,
                    typeConverter->convertType(op.getRes().getType()),
                    adaptor.getPtr()
            );
            pt::AssignOp::create(rewriter, op.getLoc(), adaptor.getPtr(), adaptor.getVal());
            return mlir::success();
        }
    };

    template< typename op_t >
    struct memcpy_insensitive : mlir::OpConversionPattern< op_t > {
        using source = op_t;
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;
        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            auto src_deref = pt::DereferenceOp::create(rewriter,
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

    using store_patterns = util::type_list<
        store_op,
        memcpy_insensitive< mlir::LLVM::MemcpyOp >,
        memcpy_insensitive< mlir::LLVM::MemmoveOp >,
        memset_insensitive
    >;

    struct va_start : mlir::OpConversionPattern< mlir::LLVM::VaStartOp > {
        using source = mlir::LLVM::VaStartOp ;
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;
        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            rewriter.replaceOpWithNewOp< pt::VaStartOp >(op, adaptor.getArgList());
            return mlir::success();
        }
    };

    struct va_copy : mlir::OpConversionPattern< mlir::LLVM::VaCopyOp > {
        using source = mlir::LLVM::VaCopyOp ;
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;
        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            auto src_deref = pt::DereferenceOp::create(
                rewriter,
                op.getLoc(),
                pt::PointerType::get(rewriter.getContext()), adaptor.getSrcList()
            );
            rewriter.replaceOpWithNewOp< pt::AssignOp >(
                 op,
                 adaptor.getDestList(),
                 src_deref
            );
            return mlir::success();
        }
    };

    struct va_arg : mlir::OpConversionPattern< mlir::LLVM::VaArgOp > {
        using source = mlir::LLVM::VaArgOp ;
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;
        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            rewriter.replaceOpWithNewOp< pt::DereferenceOp >(
                 op,
                 typeConverter->convertType(op.getRes().getType()),
                 adaptor.getArg()
            );
            return mlir::success();
        }
    };

    using vararg_patterns = util::type_list<
        va_start,
        va_arg,
        va_copy
    >;

    struct load_op : mlir::OpConversionPattern< mlir::LLVM::LoadOp > {
        using base = mlir::OpConversionPattern< mlir::LLVM::LoadOp >;
        using base::base;
        using adaptor_t = typename mlir::LLVM::LoadOp::Adaptor;

        logical_result matchAndRewrite(mlir::LLVM::LoadOp op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            rewriter.replaceOpWithNewOp< pt::DereferenceOp >(
                    op,
                    typeConverter->convertType(op.getRes().getType()),
                    adaptor.getAddr()
            );
            return mlir::success();
        }
    };

    template< typename op_t >
    struct extract_op : mlir::OpConversionPattern< op_t > {
        using base = mlir::OpConversionPattern< op_t >;
        using base::base;
        using adaptor_t = typename op_t::Adaptor;

        logical_result matchAndRewrite(op_t op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            rewriter.replaceOpWithNewOp< pt::CopyOp >(
                    op,
                    this->typeConverter->convertType(op.getRes().getType()),
                    adaptor.getOperands()[0]
            );
            return mlir::success();
        }
    };

    using load_patterns = util::type_list<
        load_op,
        // TODO: these do not load from memory, move them
        extract_op< mlir::LLVM::ExtractValueOp >,
        extract_op< mlir::LLVM::ExtractElementOp >
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
            rewriter.replaceOpWithNewOp< pt::CopyOp >(
                    op,
                    this->typeConverter->convertType(op.getType()),
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
            rewriter.replaceOpWithNewOp< pt::CopyOp >(
                        op,
                        typeConverter->convertType(op.getType()),
                        adaptor.getBase()
                );
            return mlir::success();
        }
    };

    template< typename op_t >
    struct shift_op : mlir::OpConversionPattern< op_t > {
        using source = op_t;
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;
        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            rewriter.replaceOpWithNewOp< pt::CopyOp >(
                        op,
                        this->typeConverter->convertType(op.getType()),
                        adaptor.getLhs()
                );
            return mlir::success();
        }
    };

    template< typename op_t >
    struct funnel_shift_op : mlir::OpConversionPattern< op_t > {
        using source = op_t;
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;
        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            rewriter.replaceOpWithNewOp< pt::CopyOp >(
                        op,
                        this->typeConverter->convertType(op.getType()),
                        mlir::ValueRange{adaptor.getA(), adaptor.getB()}
                );
            return mlir::success();
        }
    };

    template< typename op_t >
    struct insert_op : mlir::OpConversionPattern< op_t > {
        using source = op_t;
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;
        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            rewriter.replaceOpWithNewOp< pt::CopyOp >(
                    op,
                    this->typeConverter->convertType(op.getType()),
                    mlir::ValueRange{adaptor.getOperands()[0], adaptor.getValue()}
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
                    typeConverter->convertType(op.getType()),
                    mlir::ValueRange{adaptor.getTrueValue(), adaptor.getFalseValue()}
            );
            return mlir::success();
        }
    };

    using copy_patterns = util::type_list<
        copy_op< mlir::LLVM::AddOp >,
        copy_op< mlir::LLVM::SAddWithOverflowOp >,
        copy_op< mlir::LLVM::UAddWithOverflowOp >,
        copy_op< mlir::LLVM::FAddOp >,
        copy_op< mlir::LLVM::FMulAddOp >,
        copy_op< mlir::LLVM::FNegOp >,
        copy_op< mlir::LLVM::SubOp >,
        copy_op< mlir::LLVM::SSubWithOverflowOp >,
        copy_op< mlir::LLVM::USubSat >,
        copy_op< mlir::LLVM::USubWithOverflowOp >,
        copy_op< mlir::LLVM::FSubOp >,
        copy_op< mlir::LLVM::MulOp >,
        copy_op< mlir::LLVM::FMulOp >,
        copy_op< mlir::LLVM::MulOp >,
        copy_op< mlir::LLVM::SMulWithOverflowOp >,
        copy_op< mlir::LLVM::UMulWithOverflowOp >,
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
        copy_op< mlir::LLVM::AddrSpaceCastOp >,
        copy_op< mlir::LLVM::ZExtOp >,
        copy_op< mlir::LLVM::SExtOp >,
        copy_op< mlir::LLVM::FAbsOp >,
        copy_op< mlir::LLVM::URemOp >,
        copy_op< mlir::LLVM::SRemOp >,
        copy_op< mlir::LLVM::FRemOp >,
        copy_op< mlir::LLVM::AndOp >,
        copy_op< mlir::LLVM::OrOp >,
        copy_op< mlir::LLVM::XOrOp >,
        copy_op< mlir::LLVM::BitcastOp >,
        copy_op< mlir::LLVM::FPExtOp >,
        copy_op< mlir::LLVM::FPToSIOp >,
        copy_op< mlir::LLVM::FPToUIOp >,
        copy_op< mlir::LLVM::FPTruncOp >,
        copy_op< mlir::LLVM::FreezeOp >,
        copy_op< mlir::LLVM::AbsOp >,
        copy_op< mlir::LLVM::SMaxOp >,
        copy_op< mlir::LLVM::UMaxOp >,
        copy_op< mlir::LLVM::UMinOp >,
        copy_op< mlir::LLVM::SMinOp >,
        copy_op< mlir::LLVM::vector_reduce_add >,
        copy_op< mlir::LLVM::vector_reduce_and >,
        copy_op< mlir::LLVM::vector_reduce_or >,
        copy_op< mlir::LLVM::vector_reduce_smax >,
        copy_op< mlir::LLVM::ShuffleVectorOp >,
        copy_op< mlir::LLVM::ByteSwapOp >,
        copy_op< mlir::LLVM::BitReverseOp >,
        copy_op< mlir::LLVM::UAddSat >,
        copy_op< mlir::LLVM::SAddSat >,
        copy_op< mlir::LLVM::USubSat >,
        copy_op< mlir::LLVM::SSubSat >,
        copy_op< mlir::LLVM::USHLSat >,
        copy_op< mlir::LLVM::SSHLSat >,
        shift_op< mlir::LLVM::ShlOp >,
        shift_op< mlir::LLVM::LShrOp >,
        shift_op< mlir::LLVM::AShrOp >,
        funnel_shift_op< mlir::LLVM::FshlOp >,
        funnel_shift_op< mlir::LLVM::FshrOp >,
        gep_insensitive,
        insert_op< mlir::LLVM::InsertValueOp >,
        insert_op< mlir::LLVM::InsertElementOp >,
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
            rewriter.replaceOpWithNewOp< pt::UnknownPtrOp >(
                    op,
                    this->typeConverter->convertType(op.getType())
            );
            return mlir::success();
        }
    };

    struct inline_asm : mlir::OpConversionPattern< mlir::LLVM::InlineAsmOp > {
        using source = mlir::LLVM::InlineAsmOp;
        using base = mlir::OpConversionPattern< source >;
        using base::base;
        using adaptor_t = typename source::Adaptor;
        logical_result matchAndRewrite(source op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            if (auto res = op.getRes()) {
                rewriter.replaceOpWithNewOp< pt::UnknownPtrOp >(op, this->typeConverter->convertType(res.getType()));
            } else {
                rewriter.eraseOp(op);
            }
            return mlir::success();
        }
    };

    using unknown_patterns = util::type_list<
        unknown_op< mlir::LLVM::IntToPtrOp >,
        inline_asm
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
            if (mlir::isa< mlir::LLVM::LLVMPointerType >(op.getType())) {
                rewriter.replaceOpWithNewOp< pt::UnknownPtrOp >(
                        op,
                        this->typeConverter->convertType(op.getType())
                );
            } else {
                rewriter.replaceOpWithNewOp< pt::ConstantOp >(
                        op,
                        this->typeConverter->convertType(op.getType()));
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
            rewriter.replaceOpWithNewOp< pt::ConstantOp >(
                    op,
                    typeConverter->convertType(op.getType())
            );
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
        constant_op< mlir::LLVM::PoisonOp >,
        constant_op< mlir::LLVM::ICmpOp >,
        constant_op< mlir::LLVM::FCmpOp >,
        constant_op< mlir::LLVM::IsConstantOp >,
        constant_op< mlir::LLVM::CountLeadingZerosOp >,
        constant_op< mlir::LLVM::CountTrailingZerosOp >,
        constant_op< mlir::LLVM::CtPopOp >,
        constant_op< mlir::LLVM::UCmpOp >,
        constant_op< mlir::LLVM::SCmpOp >,
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
            auto global = pt::NamedVarOp::create(rewriter, op.getLoc(), op.getName(), false);
            auto &orig_init = adaptor.getInitializer();
            auto &glob_init = global.getInit();

            if (!orig_init.empty()) {
                rewriter.inlineRegionBefore(orig_init, glob_init, glob_init.end());
                rewriter.replaceOp(op, global);
                return mlir::success();
            }
            if (auto val_attr = op.getValue()) {
               auto guard = mlir::OpBuilder::InsertionGuard(rewriter);
               rewriter.setInsertionPointToStart(&glob_init.emplaceBlock());
               auto constant = [&]() {
                if (mlir::isa< mlir::LLVM::LLVMPointerType >(op.getType())) {
                    if (auto symbol = mlir::dyn_cast< mlir::FlatSymbolRefAttr >(val_attr.value())) {
                        return pt::AddressOp::create(rewriter,
                                op.getLoc(),
                                typeConverter->convertType(op.getGlobalType()),
                                symbol
                        ).getResult();
                    }
                    return pt::UnknownPtrOp::create(rewriter,
                            op.getLoc(),
                            typeConverter->convertType(op.getGlobalType())
                    ).getResult();
                } else {
                    return pt::ConstantOp::create(rewriter,
                            op.getLoc(),
                            typeConverter->convertType(op.getGlobalType())
                    ).getResult();
                }
               }();
               pt::YieldOp::create(rewriter, op.getLoc(), constant);
            }
            rewriter.replaceOp(op, global);
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
                    this->typeConverter->convertType(op.getRes().getType()),
                    op.getGlobalNameAttr()
            );
            return mlir::success();
        }
    };

    using named_vars_patterns = util::type_list<
        address_of_op,
        global_op
    >;

    struct func_op : mlir::OpConversionPattern< mlir::LLVM::LLVMFuncOp > {
        using base = mlir::OpConversionPattern< mlir::LLVM::LLVMFuncOp >;
        using base::base;
        using adaptor_t = typename mlir::LLVM::LLVMFuncOp::Adaptor;

        logical_result matchAndRewrite(mlir::LLVM::LLVMFuncOp op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            auto type = op.getFunctionType();
            mlir::TypeConverter::SignatureConversion result(type.getNumParams());
            mlir::SmallVector<mlir::Type, 1> newResults;
            if (failed(typeConverter->convertSignatureArgs(type.getParams(), result)) ||
                failed(typeConverter->convertTypes(op.getResultTypes(), newResults)))
                return mlir::failure();
            mlir_type res_type = newResults.empty() ? mlir::NoneType::get(op.getContext()) : newResults[0];
            auto fn_type = pt::FunctionType::get(
                    op.getContext(),
                    res_type,
                    result.getConvertedTypes(),
                    type.isVarArg()
            );
            auto new_fn = pt::FuncOp::create(rewriter,
                    op.getLoc(),
                    op.getNameAttr(),
                    fn_type,
                    op.getAllArgAttrs(),
                    op.getAllResultAttrs()
            );
            if (!op.getBody().empty()) {
                auto &new_body = new_fn.getBody();
                rewriter.inlineRegionBefore(op.getBody(), new_body, new_body.end());
                if (failed(rewriter.convertRegionTypes(&new_body, *typeConverter, &result)))
                    return mlir::failure();
            }
            rewriter.replaceOp(op, new_fn);
            return mlir::success();
        }
    };

    struct call_op : mlir::OpConversionPattern< mlir::LLVM::CallOp > {
        using base = mlir::OpConversionPattern< mlir::LLVM::CallOp >;
        using base::base;
        using adaptor_t = typename mlir::LLVM::CallOp::Adaptor;

        logical_result matchAndRewrite(mlir::LLVM::CallOp op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            mlir::SmallVector< mlir_type > result_types;
            if (mlir::failed(typeConverter->convertTypes(op.getResultTypes(), result_types)))
                return mlir::failure();
            auto callable = op.getCallableForCallee();
            if (auto callee = mlir::dyn_cast< mlir::SymbolRefAttr >(callable)) {
                rewriter.replaceOpWithNewOp< pt::CallOp >(
                        op,
                        result_types,
                        // easiest way to get it as a string ref
                        op.getCallee().value(),
                        adaptor.getCalleeOperands(),
                        op.getArgAttrs().value_or(mlir::ArrayAttr()),
                        op.getResAttrs().value_or(mlir::ArrayAttr())
                );
                return mlir::success();
            }
            if (auto callee = mlir::dyn_cast< mlir::Value >(callable)) {
                rewriter.replaceOpWithNewOp< pt::CallIndirectOp >(
                        op,
                        result_types,
                        // adaptor getter also includes the fptr value…
                        adaptor.getCalleeOperands().front(),
                        adaptor.getCalleeOperands().drop_front(),
                        op.getArgAttrs().value_or(mlir::ArrayAttr()),
                        op.getResAttrs().value_or(mlir::ArrayAttr())
                );
                return mlir::success();
            }
            return mlir::failure();
        }
    };

    struct call_intrinsic_op : mlir::OpConversionPattern< mlir::LLVM::CallIntrinsicOp > {
        using base = mlir::OpConversionPattern< mlir::LLVM::CallIntrinsicOp >;
        using base::base;
        using adaptor_t = typename mlir::LLVM::CallIntrinsicOp::Adaptor;

        logical_result matchAndRewrite(mlir::LLVM::CallIntrinsicOp op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            auto name = op.getIntrin();
            if (name.starts_with("llvm.load.relative")) {
                rewriter.replaceOpWithNewOp< pt::CopyOp >(
                        op,
                        this->typeConverter->convertType(op.getResult(0).getType()),
                        adaptor.getOperands()[0]
                    );
                return mlir::success();
            }
            return mlir::failure();
        }
    };

    using func_patterns = util::type_list<
        func_op,
        call_op,
        call_intrinsic_op,
        cf::yield_pattern< mlir::LLVM::ReturnOp >
    >;

    template< typename op_t >
    struct erase_pattern : mlir::OpConversionPattern< op_t > {
        using base = mlir::OpConversionPattern< op_t >;
        using base::base;
        using adaptor_t = typename op_t::Adaptor;

        logical_result matchAndRewrite(op_t op,
                                       adaptor_t adaptor,
                                       mlir::ConversionPatternRewriter &rewriter
        ) const override {
            rewriter.eraseOp(op);
            return mlir::success();
        }
    };

    using erase_patterns = util::type_list<
        erase_pattern< mlir::LLVM::LifetimeStartOp>,
        erase_pattern< mlir::LLVM::LifetimeEndOp>,
        erase_pattern< mlir::LLVM::VaEndOp >
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
        erase_patterns,
        func_patterns,
        load_patterns,
        named_vars_patterns,
        store_patterns,
        unknown_patterns,
        vararg_patterns
    >;

    struct LLVMIRToPoTAToPass : impl::LLVMIRToPoTAToBase< LLVMIRToPoTAToPass > {
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
            patterns.add< cf::branch_pattern >(&mctx);

            trg.addDynamicallyLegalDialect< mlir::LLVM::LLVMDialect >(
                    [&](auto *op){
                        return mlir::isa< mlir::LLVM::NoAliasScopeDeclOp,
                                          mlir::LLVM::UnreachableOp,
                                          mlir::LLVM::AssumeOp,
                                          mlir::LLVM::FenceOp,
                                          mlir::LLVM::ModuleFlagsOp
                                        > (op);
            });
            mlir::ConversionConfig cfg{};
            cfg.allowPatternRollback = false;

            if (failed(applyPartialConversion(getOperation(),
                                       trg,
                                       std::move(patterns),
                                       cfg)))
                    return signalPassFailure();
        }
    };

} // namespace potato::conv::llvmtopt

std::unique_ptr< mlir::Pass > potato::createLLVMToPotatoPass() {
    return std::make_unique< potato::conv::llvmtopt::LLVMIRToPoTAToPass >();
}

