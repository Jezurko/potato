#ifndef POTATO_PASSES
#define POTATO_PASSES

include "mlir/Pass/PassBase.td"

def PointsToPass : Pass< "points-to-pass", "mlir::ModuleOp" > {
  let summary = "Run the points to analysis.";
  let description = [{TODO}];

  let dependentDialects = [
    "potato::pt::PotatoDialect"
  ];

  let options = [
    Option<"print_lattice", "print_lattice", "bool", "true", "Toggle printing the whole lattice." >,
    Option<"print_stats", "print_stats", "bool", "false", "Toggle printing of the stats for each location." >,
    Option<"print_func_stats", "print_func_stats", "bool", "false", "Toggle printing of the stats for functions." >
  ];

  let constructor = "potato::pt::createPointsToPass()";
}

def AABenchPass : Pass< "aa-bench", "mlir::ModuleOp" > {
  let summary = "Simply run the andersen analysis pass.";
  let description = [{TODO}];

  let dependentDialects = [
    "potato::pt::PotatoDialect"
  ];

  let constructor = "potato::pt::createAABenchPass()";
}

def SteensgaardPointsToPass : Pass< "steensgaard-points-to-pass", "mlir::ModuleOp" > {
  let summary = "Run the points to analysis.";
  let description = [{TODO}];

  let dependentDialects = [
    "potato::pt::PotatoDialect"
  ];

  let options = [
    Option<"print_lattice", "print_lattice", "bool", "true", "Toggle printing the whole lattice." >,
    Option<"print_stats", "print_stats", "bool", "false", "Toggle printing of the stats for each location." >,
    Option<"print_func_stats", "print_func_stats", "bool", "false", "Toggle printing of the stats for functions." >
  ];

  let constructor = "potato::pt::createSteensgaardPointsToPass()";
}

def SteensBenchPass : Pass< "steens-bench", "mlir::ModuleOp" > {
  let summary = "Run the steensgaard analysis pass.";
  let description = [{TODO}];

  let dependentDialects = [
    "potato::pt::PotatoDialect"
  ];

  let constructor = "potato::pt::createSteensBenchPass()";
}

def LLVMPointsToPass : Pass< "llvm-points-to-pass", "mlir::ModuleOp" > {
  let summary = "Run the points to analysis.";
  let description = [{TODO}];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect"
  ];

  let options = [
    Option<"print_lattice", "print_lattice", "bool", "true", "Toggle printing the whole lattice." >,
    Option<"print_stats", "print_stats", "bool", "false", "Toggle printing of the stats for each location." >,
    Option<"print_func_stats", "print_func_stats", "bool", "false", "Toggle printing of the stats for functions." >
  ];

  let constructor = "potato::pt::createLLVMPointsToPass()";
}

#endif // POTATO_DIALECT
