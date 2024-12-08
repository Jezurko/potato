// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null


void may_alias(void *p, void *q) { return; }
void no_alias(void *p, void *q) { return; }

#include <stdarg.h>
void *foo(int c, void *a, ...) {
    va_list ap;
    void *p;
    va_start(ap, a);
    for (int j = 0; j < c; j++) {
        p = va_arg(ap, void *);
        may_alias(p, a);
    }
    va_end(ap);

    return p;
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
