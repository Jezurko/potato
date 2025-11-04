#ifndef POTATO_DIALECT_OPS
#define POTATO_DIALECT_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Potato_AddressOp
  : Potato_Op< "address", [SymbolUserOpInterface, DeclareOpInterfaceMethods<SymbolUserOpInterface>] >
  , Arguments<( ins FlatSymbolRefAttr:$symbol )>
  , Results<( outs AnyType:$ptr )>
{
  let assemblyFormat = [{ attr-dict $symbol `:` functional-type(operands, results) }];
}

def Potato_CopyOp
  : Potato_Op< "copy", [NoMemoryEffect] >
  , Arguments<( ins Variadic< AnyType >: $srcs )>
  , Results<( outs AnyType:$res )>
{
  let hasFolder = 1;

  let assemblyFormat = [{ attr-dict $srcs `:` functional-type(operands, results) }];
}

def Potato_AssignOp
  : Potato_Op< "assign" >
  , Arguments<( ins Arg< AnyType, "", [MemWrite] >: $lhs,
                    Arg< AnyType, "", [MemRead] >:  $rhs )>
{
  let hasCanonicalizeMethod = 1;

  let assemblyFormat = [{ attr-dict `*` $lhs `=` $rhs `:` type(operands) }];
}

def Potato_DereferenceOp
  : Potato_Op< "deref" >
  , Arguments<( ins AnyType: $ptr )>
  , Results<( outs AnyType: $val )>
{
  let assemblyFormat = [{ attr-dict $ptr `:` functional-type(operands, results) }];
}

def Potato_AllocOp
  : Potato_Op< "alloc" >
  , Results<( outs Res< AnyType, "", [MemAlloc] >:$result )>
{
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

def Potato_ConstantOp
  : Potato_Op< "constant", [ConstantLike, NoMemoryEffect] >
  , Results<( outs AnyType:$result )>
{
  let hasFolder = 1;

  let assemblyFormat = [{ attr-dict `:` type($result)}];
}

def Potato_NamedVarOp
  : Potato_Op< "named_var", [IsolatedFromAbove, Symbol] >
  , Arguments<( ins SymbolNameAttr:$sym_name, BoolAttr:$local )>
{
  let regions = (region AnyRegion:$init);
  let assemblyFormat = [{ attr-dict $sym_name `local` `:` $local ($init^)? }];
}

def Potato_UnknownPtrOp
  : Potato_Op< "unknown.ptr" >
  , Results<( outs AnyType:$res )>
{
  let summary = "Op that creates a pointer pointing anywhere.";
  let description = [{
      This operation is a way to introduce the "top" value of the points-to lattice into the IR.
      This operation might be used when the user has some operation that might cause some unpredictable effects (e.g. overflow)
      or when they are not interested in some of the computed information and would prefer faster convergence.
    }];

  let assemblyFormat = [{ attr-dict `:` type(results) }];
}

def Potato_BranchOp
  : Potato_Op< "br", [Terminator, BranchOpInterface, DeclareOpInterfaceMethods<BranchOpInterface>] >
{
  let arguments = (ins VariadicOfVariadic< AnyType, "succ_operand_segments" >:$succOperands,
                       DenseI32ArrayAttr:$succ_operand_segments);
  let successors = (successor VariadicSuccessor< AnySuccessor >:$succs);

  let assemblyFormat = [{ attr-dict custom<Branches>($succs, $succ_operand_segments, $succOperands, type($succOperands)) }];
}

def Potato_ValuedConstantOp
  : Potato_Op< "val_constant", [ConstantLike, NoMemoryEffect] >
  , Arguments<( ins TypedAttrInterface:$value )>
  , Results<( outs AnyType:$result )>
{
  let builders = [
    OpBuilder<(ins "mlir::Type":$type, "bool":$value), [{
      build($_builder, $_state, type, mlir::BoolAttr::get(type.getContext(), value));
    }]>,
    OpBuilder<(ins "mlir::Type":$type, "llvm::APSInt":$value), [{
      build($_builder, $_state, type, mlir::IntegerAttr::get(type.getContext(), value));
    }]>,
    OpBuilder<(ins "mlir::Type":$type, "llvm::APFloat":$value), [{
      build($_builder, $_state, type, mlir::FloatAttr::get(type, value));
    }]>
  ];

  let hasFolder = 1;

  let assemblyFormat = [{$value attr-dict `:` type($result)}];
}

def Potato_YieldOp
  : Potato_Op< "yield", [ReturnLike, Terminator] >
  , Arguments<( ins Variadic< AnyType >:$value )>
{
  let assemblyFormat = [{attr-dict ($value^ `:` type($value))?}];
}

def Potato_FuncOp
  : Potato_Op< "func", [IsolatedFromAbove, FunctionOpInterface]>
{
  let arguments = (ins
    StrAttr:$sym_name,
    TypeAttrOf< Potato_FunctionType >:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region AnyRegion:$body);
  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{

    ::llvm::ArrayRef< ::mlir::Type > getResultTypes() {
      return getFunctionType().getReturnTypes();
    }

    ::llvm::ArrayRef< ::mlir::Type > getArgumentTypes() {
      return getFunctionType().getParams();
    }

    bool isVarArg() { return getFunctionType().isVarArg(); }

    ::mlir::Region * getCallableRegion() {
      if (isExternal())
        return nullptr;
      return &getBody();
    }
  }];

  //let hasVerifier = 1;
  //let hasRegionVerifier = 1;
}

def CallOp
  : Potato_Op< "call", [CallOpInterface, DeclareOpInterfaceMethods< SymbolUserOpInterface >] > {
  let summary = "call operation";
  let description = [{
    TODO
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$operands,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs,
    UnitAttr:$no_inline
  );

  let results = (outs Variadic<AnyType>);

  let extraClassDeclaration = [{
    // Get operands passed as arguments to the called function
    operand_range getArgOperands() { return getOperands(); }

    ::mlir::MutableOperandRange getArgOperandsMutable() { return getOperandsMutable(); }

    /// Return the callee of this operation.
    ::mlir::CallInterfaceCallable getCallableForCallee() { return getCalleeAttr(); }

    /// Set the callee for this operation.
    void setCalleeFromCallable(::mlir::CallInterfaceCallable callee) {
      (*this)->setAttr("callee", ::mlir::cast< ::mlir::SymbolRefAttr >(callee));
    }
  }];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];
}

def CallIndirectOp
  : Potato_Op< "call_indirect", [CallOpInterface] > {
  let summary = "indirect call operation";
  let description = [{
    TODO
  }];

  let arguments = (ins
    AnyType:$callee,
    Variadic<AnyType>:$callee_operands,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let results = (outs Variadic<AnyType>:$results);

  let extraClassDeclaration = [{
    // Get operands passed as arguments to the called function
    operand_range getArgOperands() { return getCalleeOperands(); }

    ::mlir::MutableOperandRange getArgOperandsMutable() {
      return getCalleeOperandsMutable();
      }

    /// Return the callee of this operation.
    ::mlir::CallInterfaceCallable getCallableForCallee() { return getCallee(); }

    /// Set the callee for this operation.
    void setCalleeFromCallable(::mlir::CallInterfaceCallable callee) {
      setOperand(0, ::mlir::cast< ::mlir::Value >(callee));
    }
  }];

  let assemblyFormat = [{
    $callee `(` $callee_operands `)` attr-dict `:` type($callee) `,` functional-type($callee_operands, $results)
  }];
}

def VaStartOp : Potato_Op< "vastart", [] > {
    let summary = "Operation for intializing a memory cell with functions VarArg list.";
    let description = [{
        This operation mimics the LLVM va_start intrinsic.
        It initializes the memory at `arg_list` with a pointer to the vararg list."
    }];

    let arguments = (ins AnyType:$arg_list);
    let assemblyFormat = [{$arg_list attr-dict `:` type($arg_list)}];
}

#endif // POTATO_DIALECT_OPS
