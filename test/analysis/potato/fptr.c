// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --canonicalize="region-simplify=disabled" --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --canonicalize="region-simplify=disabled" --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

void may_alias(int *p, int *q) { return; }
void must_alias(int *p, int *q) { return; }

int x;
int y;
int z;
int *px = &x;
int *py = &y;
int *pz = &z;

void foo() { px = &y; }
void bar() { px = &z; }

int main() {
    void (*fp)();
    fp = &foo;
    fp = &bar;
    fp();
    may_alias(px, py);
    may_alias(px, pz);
    return 0;
}
