#ifndef POTATO_DIALECT_TYPES
#define POTATO_DIALECT_TYPES

class Potato_Type< string name, string _mnemonic, list< Trait > traits = [] >
  : TypeDef< Potato_Dialect, name, traits >
{
  let mnemonic = _mnemonic;
}

def Potato_Pointer : Potato_Type< "Pointer", "ptr" >
{}

def Potato_FunctionType : Potato_Type<"Function", "func"> {
  let summary = "Potato function type";
  let description = [{
    TODO
  }];

  let parameters = (ins "::mlir::Type":$returnType, ArrayRefParameter<"::mlir::Type">:$params,
                        "bool":$varArg);
  let assemblyFormat = [{
    `<` $returnType ` ` `(`
    custom<FunctionType>($params, $varArg) `>`
  }];

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::mlir::Type":$result, "::llvm::ArrayRef<::mlir::Type>":$arguments,
      CArg<"bool", "false">:$isVarArg),
      [{
        return Base::get(result.getContext(), result, arguments, isVarArg);
      }]>
  ];

  let extraClassDeclaration = [{
    /// Returns whether the function is variadic.
    bool isVarArg() const { return getVarArg(); }

    FunctionType clone(::mlir::TypeRange inputs, ::mlir::TypeRange results) const;

    /// Returns the result type of the function as an ArrayRef, enabling better
    /// integration with generic MLIR utilities.
    ::llvm::ArrayRef<::mlir::Type> getReturnTypes() const;

    /// Returns the number of arguments to the function.
    size_t getNumParams() const { return getParams().size(); }

    /// Returns `i`-th argument of the function. Asserts on out-of-bounds.
    ::mlir::Type getParamType(unsigned i) { return getParams()[i]; }
  }];
}
#endif // POTATO_DIALECT_TYPES
