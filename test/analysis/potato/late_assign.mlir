// RUN: %potato-opt %s --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null | %file-check %s

"builtin.module"() ( {
    func.func @foo() {
        %0 = pt.alloc : !pt.ptr
        %1 = pt.alloc : !pt.ptr
        %2 = pt.deref %1 : (!pt.ptr) -> !pt.ptr
        pt.assign %1 = %0 : !pt.ptr, !pt.ptr
        func.return
    }
} ): () -> ()

// CHECK-DAG: %0 = pt.alloc {{.*}} -> {[[alloc:.*]]}
// CHECK-DAG: %2 = pt.deref {{.*}} -> {[[alloc]]}
