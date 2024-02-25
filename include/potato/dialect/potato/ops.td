#ifndef POTATO_DIALECT_OPS
#define POTATO_DIALECT_OPS

include "mlir/IR/OpBase.td"

def Potato_AddressOfOp
  : Potato_Op< "address_of" >
  , Arguments<( ins AnyType:$lhs, AnyType:$rhs)>
{
  let assemblyFormat = [{ attr-dict `(` $lhs `=` `addr_of` $rhs`)` `:` functional-type(operands, results) }];
}

def Potato_CopyOp
  : Potato_Op< "copy" >
  , Arguments<( ins AnyType:$lhs, AnyType:$rhs )>
{
  let assemblyFormat = [{ attr-dict `(` $lhs `=` $rhs`)` `:` functional-type(operands, results) }];
}

def Potato_AssignOp
  : Potato_Op< "assign" >
  , Arguments<( ins AnyType:$lhs, AnyType:$rhs )>
{
  let assemblyFormat = [{ attr-dict `(` `*` $lhs `=` $rhs`)` `:` functional-type(operands, results) }];
}


def Potato_DereferenceOp
  : Potato_Op< "deref" >
  , Arguments<( ins AnyType:$lhs, AnyType:$rhs )>
{
  let assemblyFormat = [{ attr-dict `(` $lhs `=` `*` $rhs`)` `:` functional-type(operands, results) }];
}

def Potato_MAllocOp
  : Potato_Op< "malloc" >
  , Arguments<( ins )>
  , Results<( outs AnyType:$result)>
{
  let assemblyFormat = [{ attr-dict `(` `new_memory_location` `)` `:` functional-type(operands, results) }];
}

#endif // POTATO_DIALECT_OPS
