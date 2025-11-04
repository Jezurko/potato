// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

void may_alias(void *p, void *q) { return; }
void no_alias(void *p, void *q) { return; }

#include <stdarg.h>
int *foo(int c, void *a, ...) {
    int *res;
    va_list ap;
    va_start(ap, a);
    res = va_arg(ap, int *);
    va_end(ap);
    return res;
}

int x;
int y;
int z;
int main() {
    int *p = foo(2, &z, &x, &y);
    may_alias(p, &x);
    may_alias(p, &y);
    no_alias(p, &z);
    return 0;
}
