/*
 * Heap
 * Author: Sen Ye
 * Date: 12/10/2013
 * Description: heap objects are identified according to their
 *		allocation sites.
 */

// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -Wno-implicit-function-declaration -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

#include "aliascheck.h"

// return two malloc object
void malloc_two(int **p, int **q) {
	*p = (int*) malloc(sizeof(int));
	*q = (int*) malloc(sizeof(int));
}

int main() {
	int **o1 = malloc(100);
    int **o2 = malloc(100);
	malloc_two(o1, o2);
	no_alias(*o1, *o2);
	return 0;
}
