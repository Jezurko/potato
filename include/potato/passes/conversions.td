#ifndef POTATO_CONVERSION_PASSES_TD
#define POTATO_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def LLVMIRToPoTATo : Pass<"llvm-ir-to-potato", "mlir::ModuleOp"> {
  let summary = "LLVM IR to PoTATo dialect converison";
  let description = [{
    TODO
  }];

  let constructor = "potato::createLLVMToPotatoPass()";
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "potato::pt::PotatoDialect"
  ];
}


#endif // POTATO_CONVERSION_PASSES_TD

