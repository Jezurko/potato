#ifnddef POTATO_DIALECT_OPS
#define POTATO_DIALECT_OPS

include "mlir/IR/OpBase.td"

def Potato_AddressOfOp
  : Potato_Op< "address_of" >
  , Arguments< ins AnyType:$ptset, AnyType:$var, AnyType:$address >
  , Results< outs AnyType:$ptset >
{
  let assemblyFormat [{ $ptset `<-` `(` $var `:` $address `)` `:` functional-type(operands, results)}]
}

def Potato_CopyOp
  : Potato_Op< "copy" >
  , Arguments< ins AnyType:$ptset, AnyType:$lhs, AnyType:$rhs >
  , Results< outs AnyType:$ptset >
{
  let assemblyFormat [{ $ptset `<-` `(` $lhs `=` $rhs`)` `:` functional-type(operands, results)}]

}

def Potato_AssignOp
  : Potato_Op< "assign" >
  , Arguments< ins AnyType:$ptset, AnyType:$lhs, AnyType:$rhs >
  , Results< outs AnyType:$ptset >
{
  let assemblyFormat [{ $ptset `<-` `(` `*`$lhs `=` $rhs`)` `:` functional-type(operands, results)}]
}


def Potato_DereferenceOp
  : Potato_Op< "deref" >
  , Arguments< ins AnyType:$ptset, AnyType:$lhs, AnyType:$rhs >
  , Results< outs AnyType:$ptset >
{
  let assemblyFormat [{ $ptset `<-` `(` $lhs `=` `*`$rhs`)` `:` functional-type(operands, results)}]
}

def Potato_MAllocOp
  : Potato_Op< "malloc" >
  , Arguemnts< ins AnyType:$ptset, AnyType:$var>
  , Results< outs AnyType:$ptset >
{
  let assemblyFormat [{ $ptset `<-` `(` $var `:` `new_var` `)` `:` functional-type(operands, results) }]

}

#endif // POTATO_DIALECT_OPS
