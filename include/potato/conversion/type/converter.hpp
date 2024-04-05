#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/Types.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/potato.hpp"
#include "potato/util/common.hpp"

namespace potato::conv {
    struct to_pt_type : mlir::TypeConverter {
        using base = mlir::TypeConverter;
        using base::base;

        to_pt_type() : base() {
            addConversion([&](mlir_type t) -> mlir_type {
                return pt::PointerType::get(t.getContext());
            });

            auto materializer =
                [&](mlir::OpBuilder &builder, mlir_type resultType,
                    mlir::ValueRange inputs, mlir_loc loc) -> std::optional< mlir_value >
                {
                    if (inputs.size() != 1) {
                        return std::nullopt;
                    }

                    return builder
                        .create< mlir::UnrealizedConversionCastOp >(loc, resultType, inputs)
                        .getResult(0);
                };

            addTargetMaterialization(materializer);
            addSourceMaterialization(materializer);
            addArgumentMaterialization(materializer);

        }
    };

    // Of `self_t` only `getTypeConverter` is required.
    template< typename self_t, typename type_converter >
    struct do_type_conversion_on_op
    {
      private:
        const auto &self() const { return static_cast< const self_t & >(*this); }

      public:

        auto &get_type_converter() const {
            return static_cast< type_converter & >(*self().getTypeConverter());
        }

        logical_result replace(mlir::FunctionOpInterface fn,
                               auto &rewriter) const
        {
            auto old_type = fn.getFunctionType();
            auto trg_type = get_type_converter().convertFunctionType(old_type);
            old_type.dump();
            trg_type.dump();
            auto update = [&]() {
                fn.setType(trg_type);
                if (!fn.empty() && fn->getNumRegions() != 0) {
                    fixup_entry_block(fn.front());
                }
            };

            rewriter.updateRootInPlace(fn, update);
            return mlir::success();
        }

        logical_result replace(
            mlir::Operation *op,
            auto &rewriter
        ) const {
            auto &tc = get_type_converter();

            auto update = [&]() {
                mlir::AttrTypeReplacer replacer;

                replacer.addReplacement([&](mlir_type t) { return tc.convertType(t); }
                );

                replacer.recursivelyReplaceElementsIn(
                    op
                    , true /* replace attrs */
                    , false /* replace locs */
                    , true /* replace types */
                );

                if (op->getNumRegions() != 0) {
                    fixup_entry_block(op->getRegion(0));
                }
            };

            rewriter.updateRootInPlace(op, update);

            return mlir::success();
        }

        void fixup_entry_block(mlir::Block &block) const {
            for (auto arg : block.getArguments()) {
                auto trg = get_type_converter().convertType(arg.getType());
                arg.setType(trg);
            }
        }

        void fixup_entry_block(mlir::Region &region) const {
            if (region.empty()) {
                return;
            }

            return fixup_entry_block(region.front());
        }
    };

    template< typename type_converter >
    struct generic_type_converting_pattern
        : mlir::ConversionPattern,
          do_type_conversion_on_op< generic_type_converting_pattern< type_converter >,
                                    type_converter >
    {
        using base = mlir::ConversionPattern;
        using base::base;

        generic_type_converting_pattern(mlir::TypeConverter &tc,
                        mlir::MLIRContext &mctx)
            : base(tc, mlir::Pattern::MatchAnyOpTypeTag{}, 1, &mctx)
        {}

        logical_result matchAndRewrite(
            mlir_operation *op, mlir::ArrayRef< mlir::Value >,
            mlir::ConversionPatternRewriter &rewriter
        ) const override {
            if (auto func_op = mlir::dyn_cast< mlir::FunctionOpInterface >(op))
                return this->replace(func_op, rewriter);
            return this->replace(op, rewriter);
        }
    };
} // namespace potato::conv

