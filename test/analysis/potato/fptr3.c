// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --canonicalize="region-simplify=disabled" --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --canonicalize="region-simplify=disabled" --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

void may_alias(int *p, int *q) { return; }
void must_alias(int *p, int *q) { return; }

int x;
int *px = &x;
int *px2;
int *px3;

void foo() { px2 = &x; }
void bar() { px3 = &x; }

void caller(void (*fp)()) {
    fp();
}

int main() {
    void (*fp)();
    fp = &foo;
    fp = &bar;
    caller(fp);
    must_alias(px, px2);
    must_alias(px, px3);
    return 0;
}
