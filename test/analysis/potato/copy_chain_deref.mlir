// RUN: %potato-opt %s --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %potato-opt %s --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %potato-opt %s --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %potato-opt %s --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

pt.func @may_alias(%p : !pt.ptr, %q : !pt.ptr) -> none {
    pt.yield
}

pt.func @foo() -> none {
    %0 = pt.alloc : !pt.ptr
    %1 = pt.alloc : !pt.ptr
    %2 = pt.copy %0 : (!pt.ptr) -> !pt.ptr
    %3 = pt.copy %2 : (!pt.ptr) -> !pt.ptr
    %4 = pt.deref %3 : (!pt.ptr) -> !pt.ptr
    pt.assign * %3 = %1 : !pt.ptr, !pt.ptr
    pt.call @may_alias(%4, %1)  : (!pt.ptr, !pt.ptr) -> ()
    pt.yield
}
