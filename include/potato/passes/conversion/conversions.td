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
    "potato::pt::PotatoDialect"
  ];
}

def FunctionModelling : Pass<"add-fn-bodies-from-models", "mlir::ModuleOp"> {
  let summary = "Pass that adds bodies to functions specified by models.";
  let description = [{
    TODO
  }];

  let options = [
    Option<"override_existing", "override-existing-bodies", "bool", /*default=*/"false",
           "If set to true the pass will overwrite bodies of defined functions when a model is available.">,
    Option<"inline_bodies", "inline-model-bodies", "bool", /*default=*/"true",
           "If set to true the pass will inline the bodies of models when possible.">,
  ];

  let constructor = "potato::createFunctionModellingPass()";
  let dependentDialects = [
    "potato::pt::PotatoDialect"
  ];
}


#endif // POTATO_CONVERSION_PASSES_TD

