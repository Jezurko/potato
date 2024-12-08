// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

void may_alias(int *p, int *q) { return; }
void must_alias(int *p, int *q) { return; }
void no_alias(int *p, int *q) { return; }

int x;
int y;

int *foo() { return &x; }
int *bar() { return &y; }

int *caller1(int *(*fp)()) {
    return fp();
}

int *caller2(int *(*fp)()) {
    return fp();
}

int main() {
    int *p1 = caller1(&foo);
    int *p2 = caller2(&bar);
    no_alias(p1, p2);
    return 0;
}
