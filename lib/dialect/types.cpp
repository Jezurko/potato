#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/potato.hpp"
#include "potato/dialect/types.hpp"
#include "potato/util/common.hpp"

namespace potato::pt {
    void PotatoDialect::registerTypes() {
        addTypes<
            #define GET_TYPEDEF_LIST
            #include "potato/dialect/PotatoTypes.cpp.inc"
        >();
    }

    static mlir::ParseResult parseFunctionType(
            mlir::AsmParser &p, llvm::SmallVector< mlir_type > &params, bool &isVarArg)
    {
      isVarArg = false;
      // `(` `)`
      if (succeeded(p.parseOptionalRParen()))
        return mlir::success();

      // `(` `...` `)`
      if (succeeded(p.parseOptionalEllipsis())) {
        isVarArg = true;
        return p.parseRParen();
      }

      // type (`,` type)* (`,` `...`)?
      mlir::Type type;
      if (p.parseType(type))
        return mlir::failure();
      params.push_back(type);
      while (succeeded(p.parseOptionalComma())) {
        if (succeeded(p.parseOptionalEllipsis())) {
          isVarArg = true;
          return p.parseRParen();
        }
        if (p.parseType(type))
          return mlir::failure();
        params.push_back(type);
      }
      return p.parseRParen();
    }

    static void printFunctionType(mlir::AsmPrinter &p, llvm::ArrayRef<mlir_type> params,
                                   bool isVarArg) {
      llvm::interleaveComma(params, p,
                            [&](mlir_type type) { p.printType(type); });
      if (isVarArg) {
        if (!params.empty())
          p << ", ";
        p << "...";
      }
      p << ')';
    }
} // namespace potato::pt

#define GET_TYPEDEF_CLASSES
#include "potato/dialect/PotatoTypes.cpp.inc"

namespace potato::pt {
    FunctionType FunctionType::clone(
            mlir::TypeRange inputs, mlir::TypeRange results
    ) const {
      if (results.size() != 1)
        return {};
      return get(results[0], llvm::to_vector(inputs), isVarArg());
    }


    llvm::ArrayRef< mlir_type > FunctionType::getReturnTypes() const {
      return static_cast< detail::FunctionTypeStorage * >(getImpl())->returnType;
    }
} // namespace potato::pt
