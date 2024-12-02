// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

#include "aliascheck.h"

struct innerStruct{
int m;
int* n;
};
struct myStruct{
float a;
struct innerStruct b;
};

int main(){
  struct myStruct x;
  x.b.n = malloc(10);
  struct myStruct y;
  memcpy(&y,&x,sizeof(struct myStruct));
  may_alias(x.b.n,y.b.n);
}
