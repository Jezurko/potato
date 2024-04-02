#include "potato/util/warnings.hpp"

#include <mlir/IR/DialectImplementation.h>

#include "potato/dialect/potato.hpp"
#include "potato/dialect/ops.hpp"
#include "potato/dialect/types.hpp"

namespace potato::pt
{
    void PotatoDialect::initialize()
    {
        registerTypes();
        //registerAttributes();

        addOperations<
            #define GET_OP_LIST
            #include "potato/dialect/Potato.cpp.inc"
        >();

        //addInterfaces< potatoOpDialectInterface >();

    }

    using DialectParser = mlir::AsmParser;
    using DialectPrinter = mlir::AsmPrinter;

    mlir::Operation *PotatoDialect::materializeConstant(mlir::OpBuilder &builder, mlir::Attribute value, mlir::Type type, mlir::Location loc)
    {
        assert(false);
    }

}

