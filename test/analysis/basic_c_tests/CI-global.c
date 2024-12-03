/*
 * Context-insensitive.
 * Author: Sen Ye
 * Date: 10/10/2013
 */

// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

#include "aliascheck.h"

int global;
int *p_global;

void foo() {
	p_global = &global;
}

int main() {
	int *p_local;
	p_local = &global;
	foo();
	must_alias(p_local, p_global);
	return 0;
}
