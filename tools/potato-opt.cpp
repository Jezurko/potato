#include "potato/util/warnings.hpp"

POTATO_RELAX_WARNINGS
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
POTATO_UNRELAX_WARNINGS

#include "potato/dialect/potato.hpp"
#include "potato/passes/analysis.hpp"
#include "potato/passes/conversions.hpp"

int main(int argc, char **argv)
{
    mlir::registerAllPasses();
    // Register potato passes here
    // Analysis passes
    potato::pt::registerPasses();
    // Conversion passes
    potato::registerPasses();

    mlir::DialectRegistry registry;
    // register dialects
    registry.insert< potato::pt::PotatoDialect >();
    // POTATO register
    mlir::registerAllDialects(registry);

    return mlir::failed(
        mlir::MlirOptMain(argc, argv, "PoTATo driver\n", registry)
    );
}
