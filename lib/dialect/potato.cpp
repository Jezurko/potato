#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <mlir/IR/DialectImplementation.h>
#include <mlir/Interfaces/FunctionInterfaces.h>
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/potato.hpp"
#include "potato/dialect/ops.hpp"
#include "potato/dialect/types.hpp"
#include "potato/util/common.hpp"

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


    logical_result AddressOp::verifySymbolUses(mlir::SymbolTableCollection &symbolTable) {
        mlir_operation *module = (*this)->getParentOp();
        while (module && !(module->hasTrait< mlir::OpTrait::SymbolTable >() &&
                           module->hasTrait< mlir::OpTrait::IsIsolatedFromAbove >()))
        {
            module = module->getParentOp();
        }
        assert(module && "unexpected operation outside of a module");
        mlir_operation *symbol = symbolTable.lookupSymbolIn(module, getSymbolAttr());

        if (!symbol) {
            return emitOpError("Invalid symbol.");
        }
        return logical_result::success();
    }


}

