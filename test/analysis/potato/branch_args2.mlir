// RUN: %potato-opt %s --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %potato-opt %s --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %potato-opt %s --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %potato-opt %s --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

"builtin.module"() ( {
    func.func @may_alias(%p : !pt.ptr, %q : !pt.ptr) {
        func.return
    }
    func.func @foo() {
        %0 = pt.alloc : !pt.ptr
        %a = pt.copy %0: (!pt.ptr) -> !pt.ptr
        %1 = pt.alloc : !pt.ptr
        %b = pt.copy %1 : (!pt.ptr) -> !pt.ptr
        %2 = builtin.unrealized_conversion_cast %1 : !pt.ptr to i1
        %3 = pt.alloc : !pt.ptr
        pt.br ^bb1(%0, %a : !pt.ptr, !pt.ptr), ^bb2(%1, %b : !pt.ptr, !pt.ptr)
        ^bb1(%arg0: !pt.ptr, %arg0b : !pt.ptr):
            func.call @may_alias(%arg0, %arg0b) : (!pt.ptr, !pt.ptr) -> ()
            func.return
        ^bb2(%arg1 : !pt.ptr, %arg1b : !pt.ptr):
            func.call @may_alias(%arg1, %arg1b) : (!pt.ptr, !pt.ptr) -> ()
            func.return
    }
} ): () -> ()
