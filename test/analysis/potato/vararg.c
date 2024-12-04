// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --canonicalize="region-simplify=disabled" --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --canonicalize="region-simplify=disabled" --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null


void may_alias(void *p, void *q) { return; }
void no_alias(void *p, void *q) { return; }

#include <stdarg.h>
void *foo(int c, void *a, ...) {
    return a;
}

int x;
int y;
int z;
int main() {
    void *p = foo(2, &x, &y);
    may_alias(p, &x);
    may_alias(p, &y);
    no_alias(p, &z);
    return 0;
}