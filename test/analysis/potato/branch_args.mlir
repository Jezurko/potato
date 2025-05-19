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
        %1 = pt.alloc : !pt.ptr
        %2 = builtin.unrealized_conversion_cast %1 : !pt.ptr to i1
        %3 = pt.alloc : !pt.ptr
        cf.cond_br %2, ^bb1(%0 : !pt.ptr), ^bb2(%1 : !pt.ptr)
        ^bb1(%arg0 : !pt.ptr):
            pt.assign * %3 = %0 : !pt.ptr, !pt.ptr
            cf.br ^bb3(%arg0 : !pt.ptr)
        ^bb2(%arg1 : !pt.ptr):
            pt.assign * %3 = %1 : !pt.ptr, !pt.ptr
            cf.br ^bb3(%arg1 : !pt.ptr)
        ^bb3(%arg2 : !pt.ptr):
            func.call @may_alias(%arg2, %0) : (!pt.ptr, !pt.ptr) -> ()
            func.call @may_alias(%arg2, %1) : (!pt.ptr, !pt.ptr) -> ()
            func.return
    }
} ): () -> ()
