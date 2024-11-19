/*
 * Alias due to lack of path-sensitivity.
 * Author: Sen Ye
 * Date: 06/09/2013
 */

// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

#include "aliascheck.h"

int main()
{
	int *p, *q;
	int a, b, c;
	if (c) {
		p = &a;
		q = &b;
	}
	else {
		p = &b;
		q = &c;
	}
	may_alias(p,q);
	return 0;
}