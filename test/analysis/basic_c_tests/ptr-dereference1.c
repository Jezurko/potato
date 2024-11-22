/*
 * Simple alias check
 * Author: Sen Ye
 * Date: 06/09/2013
 */

// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

#include "aliascheck.h"

int main()
{
	int a,b,*c,*d;
	c = &a;
	d = &a;
	may_alias(c,d);
	c = &b;
	// In LLVM, every declared variable is address-taken
	// accessed via pointers through loads/stores
	// c here is loaded from the same memory on LLVM's partial SSA form
	may_alias(c,d);
	no_alias(&b,d);
	return 0;
}
