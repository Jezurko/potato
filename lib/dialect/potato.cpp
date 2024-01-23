#include "potato/util/warnings.hpp"

#include <mlir/IR/DialectImplementation.h>

POTATO_RELAX_WARNINGS
#include "potato/dialect/potato/potato.hpp"
#include "potato/dialect/potato/ops.hpp"
POTATO_UNRELAX_WARNINGS

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
