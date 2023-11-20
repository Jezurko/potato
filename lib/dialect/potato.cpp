#include "potato/dialect/potato/potato.hpp"
#include "potato/dialect/potato/ops.hpp"

namespace potato::pt
{
    void potato::initialize()
    {
        //registerTypes();
        //registerAttributes();

        addOperation<
            #define GET_OP_LIST
            #include "potato/dialect/potato/potato.cpp.inc"
        >();

        addInterfaces< potatoOpDialectInterface >();

    }
}

#include "potato/dialect/potato/potato.cpp.inc"
