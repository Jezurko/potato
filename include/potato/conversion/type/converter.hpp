#pragma once

#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/Types.h>
#include <mlir/Transforms/DialectConversion.h>
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/potato.hpp"
#include "potato/util/common.hpp"

namespace potato::conv {
    struct to_pt_type : mlir::TypeConverter {
        using base = mlir::TypeConverter;
        using base::base;

        to_pt_type() : base() {
            addConversion([&](mlir_type t) {
                return pt::PointerType::get(t.getContext());
            });
        }
    };
} // namespace potato::conv

