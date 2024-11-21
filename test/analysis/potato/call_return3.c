// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --canonicalize="region-simplify=disabled" --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --canonicalize="region-simplify=disabled" --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

void may_alias(int *p, int *q) { return; }

int *fun(int *a) {
    return a;
}

int main() {
    int a, b;
    int *p = fun(&a);
    int *q = fun(&b);
    // context insensitive analysis
    may_alias(p, q);
    return 0;
}
