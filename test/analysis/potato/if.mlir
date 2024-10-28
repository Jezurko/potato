// RUN: %potato-opt %s --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null | %file-check %s
"builtin.module"() ( {
    func.func @foo() {
        %0 = pt.alloc : !pt.ptr
        %1 = pt.alloc : !pt.ptr
        %2 = builtin.unrealized_conversion_cast %1 : !pt.ptr to i1
        %3 = pt.alloc : !pt.ptr
        llvm.cond_br %2, ^bb1, ^bb2
        ^bb1:
            pt.assign %3 = %0 : !pt.ptr, !pt.ptr
            llvm.br ^bb3
        ^bb2:
            pt.assign %3 = %1 : !pt.ptr, !pt.ptr
            llvm.br ^bb3
        ^bb3:
            func.return
    }
} ): () -> ()

// CHECK: State in: {{.*}}:16:{{[0-9]+}}
// CHECK-NOT: State
// CHECK: mem_alloc2 -> {mem_alloc{{[0,1]}}, mem_alloc{{[0,1]}}}