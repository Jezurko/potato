#ifndef POTATO_PASSES
#define POTATO_PASSES

include "mlir/Pass/PassBase.td"

def PointsToPass : Pass< "points-to-pass", "mlir::ModuleOp" > {
  let summary = "Run the points to analysis.";
  let description = [{TODO}];

  let dependentDialects = [
    "potato::pt::PotatoDialect"
  ];

  let constructor = "potato::pt::createPointsToPass()";
}

#endif // POTATO_DIALECT
