#include "potato/util/warnings.hpp"

#include <mlir/IR/DialectImplementation.h>

#include "potato/dialect/potato/potato.hpp"
#include "potato/dialect/potato/ops.hpp"

namespace potato::pt
{
    void PotatoDialect::initialize()
    {
        //registerTypes();
        //registerAttributes();

        addOperations<
            #define GET_OP_LIST
            #include "potato/dialect/potato/Potato.cpp.inc"
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

