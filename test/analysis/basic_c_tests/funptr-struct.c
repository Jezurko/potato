// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %emit-llvm -o - %s | %llvm-to-mlir -o - | %potato-opt --llvm-ir-to-potato --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

#include "aliascheck.h"

int g;
static int my_sn_write(int* p) {
	  must_alias(&g,p);
	    return 0;
}

struct MYFILE {
	  int (*pt) (int* p);
};

void my_vfprintf(struct MYFILE *pts) {
	  int *p = &g;
	    pts->pt(p);
}

int my_vsnprintf() {
	  struct MYFILE pts = { .pt = my_sn_write };
	    my_vfprintf(&pts);
	      return 0;
}

int main() {
	  my_vsnprintf();
	    return 0;
}

