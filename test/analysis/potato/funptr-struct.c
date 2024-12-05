// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

void may_alias(int *p, int *q) { return; };

int x;
int y;
int *px = &x;
int *py = &y;

struct my_struct {
    int *a;
    int *(*pt) ();
    int *b;
};

int *get_px() {
    return &x;
}

int *get_py() {
    return &y;
}

int main() {
    int a, b;
    struct my_struct S = {&a, 0, &b};
    S.pt = get_px;
    int c;
    S.a = &c;
    S.pt = get_py;
    may_alias(&x, S.pt());
    may_alias(&y, S.pt());
    return 0;
}

