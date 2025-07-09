// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --add-fn-bodies-from-models --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --add-fn-bodies-from-models --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --add-fn-bodies-from-models --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --add-fn-bodies-from-models --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --add-fn-bodies-from-models="inline-model-bodies=false" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --add-fn-bodies-from-models="inline-model-bodies=false" --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --add-fn-bodies-from-models="inline-model-bodies=false" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --add-fn-bodies-from-models="inline-model-bodies=false" --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

#include <stddef.h>

size_t strlen(const char *str);
char *strcpy(char *dst, const char *src);
void may_alias(void *p, void *q) { return; }

int main(void)
{
    const char *src = "Hello.";
    char dst[7];
    char * res = strcpy(dst, src);
    may_alias(res, dst);
}
