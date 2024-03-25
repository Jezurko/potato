#ifndef POTATO_DIALECT_OPS
#define POTATO_DIALECT_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

def Potato_AddressOfOp
  : Potato_Op< "address_of" >
  , Arguments<( ins AnyType:$val )>
  , Results<( outs AnyType:$ptr )>
{
  let assemblyFormat = [{ attr-dict $val `:` functional-type(operands, results) }];
}

def Potato_CopyOp
  : Potato_Op< "copy" >
  , Arguments<( ins AnyType:$lhs, AnyType:$rhs )>
{
  let assemblyFormat = [{ attr-dict $lhs `=` $rhs `:` type(operands) }];
}

def Potato_AssignOp
  : Potato_Op< "assign" >
  , Arguments<( ins AnyType:$lhs, AnyType:$rhs )>
{
  let assemblyFormat = [{ attr-dict `*` $lhs `=` $rhs `:` type(operands) }];
}

def Potato_DereferenceOp
  : Potato_Op< "deref" >
  , Arguments<( ins AnyType:$ptr )>
  , Results<( outs AnyType:$val )>
{
  let assemblyFormat = [{ attr-dict `*` $ptr `:` functional-type(operands, results) }];
}

def Potato_AllocOp
  : Potato_Op< "alloc" >
  , Results<( outs AnyType:$result )>
{
  let assemblyFormat = [{ attr-dict `:` type($result) }];
}

def Potato_ConstantOp
  : Potato_Op< "constant", [ConstantLike] >
  , Results<( outs AnyType:$result )>
{
  let assemblyFormat = [{ attr-dict `:` type($result)}];
}

def Potato_ValuedConstantOp
  : Potato_Op< "val_constant", [ConstantLike] >
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
  let assemblyFormat = [{$value attr-dict `:` type($result)}];
}

#endif // POTATO_DIALECT_OPS
