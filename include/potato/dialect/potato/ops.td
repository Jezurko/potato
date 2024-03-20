#ifndef POTATO_DIALECT_OPS
#define POTATO_DIALECT_OPS

include "mlir/IR/OpBase.td"

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

#endif // POTATO_DIALECT_OPS
