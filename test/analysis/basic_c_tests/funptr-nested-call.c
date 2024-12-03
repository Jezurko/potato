// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

#include "aliascheck.h"

int *px;
int *qx;
int x;
void f() { px = &x; }
void g() { qx = &x; }
void (*p)();

void fake_fun (void (*a)()) {
  p = a;
  p();
}

void real_fun (void (*a)()) {
  p = a;
  p();
}

void (*fptr)(void (*p)());

void set(void (*src)()) {
  fptr = src;
}

int main(int argc, char **argv)
{
  set(&fake_fun);
  set(&real_fun);

  fptr(&f);

  fptr(&g);
  must_alias(px, qx);

  return 0;
}
