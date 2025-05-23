#ifndef POTATO_DIALECT_OPS
#define POTATO_DIALECT_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
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

  let assemblyFormat = [{ attr-dict `:` custom<Branches>($succs, $succ_operand_segments, $succOperands, type($succOperands)) }];
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

#endif // POTATO_DIALECT_OPS
