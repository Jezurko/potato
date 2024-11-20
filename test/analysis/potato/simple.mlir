// RUN: %potato-opt %s --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null | %file-check %s

"builtin.module"() ( {
    func.func @foo() {
        %0 = pt.constant : !pt.ptr
        %1 = pt.alloc : !pt.ptr
        pt.assign %1 = %0 : !pt.ptr, !pt.ptr
        %2 = pt.deref %1 : (!pt.ptr) -> !pt.ptr
        pt.assign %1 = %2 : !pt.ptr, !pt.ptr
        func.return
    }
} ): () -> ()

// CHECK-DAG: %0 = pt.constant : !pt.ptr -> {}
// CHECK-DAG: %1 = pt.alloc : !pt.ptr -> {mem_alloc at: loc({{.*}}:6:14)}
// CHECK-DAG: %2 = pt.deref %1 : (!pt.ptr) -> !pt.ptr -> {}
