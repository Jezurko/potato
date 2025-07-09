/*
 * functionpointer1.c
 * Test function pointer with field initialization of globals
 *
 *  Created on: 01/09/2015
 *      Author: Yulei Sui
 */

// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --add-fn-bodies-from-models --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --add-fn-bodies-from-models --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --add-fn-bodies-from-models --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --add-fn-bodies-from-models --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --add-fn-bodies-from-models="inline-model-bodies=false" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --add-fn-bodies-from-models="inline-model-bodies=false" --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --add-fn-bodies-from-models="inline-model-bodies=false" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --add-fn-bodies-from-models="inline-model-bodies=false" --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

#include "aliascheck.h"

typedef int PRSize;
typedef unsigned int PRUint32;
typedef unsigned int PRUintn;
typedef int PRIntn;

struct PLHashAllocOps {
   void *(*allocTable)(void *pool , PRSize size ) ;
};
typedef struct PLHashAllocOps PLHashAllocOps;

static void *DefaultAllocTable(void *pool , PRSize size )
{ void *tmp ;

  {
  tmp = malloc((unsigned int )size);
  return (tmp);
}
}

PLHashAllocOps defaultHashAllocOps  =    {& DefaultAllocTable};
void PL_NewHashTable(PRUint32 n , void *allocPriv )
{
  void *tmp___0 ;
  void *tmp___1 ;

  PLHashAllocOps const   * allocOps = (PLHashAllocOps const   *)(& defaultHashAllocOps);
  tmp___0 = (*(allocOps->allocTable))(allocPriv, (int )sizeof(int));
  tmp___1 = (*(allocOps->allocTable))(allocPriv, (int )sizeof(int));
  may_alias(tmp___0,tmp___1);

}

int main(){return 0;}

