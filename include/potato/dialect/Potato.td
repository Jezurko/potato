#ifndef POTATO_DIALECT
#define POTATO_DIALECT

include "mlir/IR/OpBase.td"

def Potato_Dialect : Dialect {
  let name = "pt";
  let cppNamespace = "::potato::pt";

  let summary = "Dialect for representing points to analysis operations.";

  let extraClassDeclaration = [{
        void registerTypes();
  }];


  let useDefaultTypePrinterParser = 1;
  //let useDefaultAttributePrinterParser = 1;

  let hasConstantMaterializer = 1;

}

class Potato_Op< string mnemonic, list< Trait > traits = [] >
  : Op< Potato_Dialect, mnemonic, traits >;

include "types.td"
include "ops.td"

#endif // POTATO_DIALECT
