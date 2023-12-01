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
}

#include "potato/dialect/potato/PotatoDialect.cpp.inc"
