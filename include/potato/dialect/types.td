#ifndef POTATO_DIALECT_TYPES
#define POTATO_DIALECT_TYPES

class Potato_Type< string name, string _mnemonic, list< Trait > traits = [] >
  : TypeDef< Potato_Dialect, name, traits >
{
  let mnemonic = _mnemonic;
}

def Potato_Pointer : Potato_Type< "Pointer", "ptr" >
{}

def Potato_Constant : Potato_Type< "Constant", "const" >
{}
#endif // POTATO_DIALECT_TYPES
