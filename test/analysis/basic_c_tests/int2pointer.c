/*
 * Alias due to casting between
 * integer and pointer.
 * Author: Sen Ye
 * Date: 06/09/2013
 */

// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

#include "aliascheck.h"

struct MyStruct {
	int f1;
	int f2;
};

int main() {
	struct MyStruct ms;
	int *p, *q;
	p = &(ms.f1);
	// cast pointer to integer
	int addr = (int)p;
	// cast integer to pointer and
	// q would point to blackhole
	q = (int*)addr + 1;
	may_alias(p, q);
	return 0;
}
