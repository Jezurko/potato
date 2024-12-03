/*
 * Cycle
 * Author: Sen Ye
 * Date: 11/10/2013
 */

// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

#include "aliascheck.h"

int main() {
	int **x1, **y1, **z1;
	int *x2, *y2, *z2, *y2_;
	int x3, y3, z3, y3_;
	x2 = &x3, y2 = &y3, z2 = &z3;
	x1 = &x2, y1 = &y2, z1 = &z2;
	// if the following branch is commented out,
	// the first alias check will fail while
	// the second one is OK.
	if (y3_) {
		y1 = &y2_;
		y2_ = &y3_;
	}
	*x1 = *y1;
	*y1 = *z1;
	*z1 = *x1;
	// there should be a cycle from
	// y2 -> x2 -> z2 -> y2
	may_alias(x2, y2);
	may_alias(z2, x2);
	return 0;
}
