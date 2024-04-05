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
} // namespace potato::conv

