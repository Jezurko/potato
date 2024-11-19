/*
 * Global variables test.
 * Author: Sen Ye
 * Date: 03/05/2014
 */

// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

#include "aliascheck.h"
#define NULL ((void *) 0)

int *p = NULL;
int *q = NULL;
int c;

void foo() {
	may_alias(p, q);
}

void bar() {
	q = &c;
}

int main() {
	int a, b;
	p = &a;
	q = p;
	p = &c;
    bar();
}
