#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/IR/DialectImplementation.h>
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/potato.hpp"
#include "potato/dialect/types.hpp"

namespace potato::pt {
    void PotatoDialect::registerTypes() {
        addTypes<
            #define GET_TYPEDEF_LIST
            #include "potato/dialect/PotatoTypes.cpp.inc"
        >();
    }
}

#define GET_TYPEDEF_CLASSES
#include "potato/dialect/PotatoTypes.cpp.inc"

