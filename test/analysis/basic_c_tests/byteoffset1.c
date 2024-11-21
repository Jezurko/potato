/// Test case from https://github.com/SVF-tools/SVF/issues/524
/// Compile this c file using `clang -S -emit-llvm -O3 byteoffset1.c`

// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --canonicalize="region-simplify=disabled" --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --canonicalize="region-simplify=disabled" --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

void must_alias(void* p, void* q){
    return;
}

typedef struct StructA
{
    int foo;
    int (*f)(void);
} StructA;

__attribute__((noinline))
int FuncA() {
    return 1;
}

__attribute__((noinline))
int CallF(StructA *structA) {
    int ret = structA->f();
    must_alias(structA->f, &FuncA);
    return ret;
}

int main() {
    StructA *structA = malloc(sizeof(StructA));
    structA->f = FuncA;

    int ret = CallF(structA);
    return ret;
}
