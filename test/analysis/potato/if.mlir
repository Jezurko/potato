// RUN: %potato-opt %s --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null | %file-check %s

"builtin.module"() ( {
    func.func @foo() {
        %0 = pt.alloc : !pt.ptr
        %1 = pt.alloc : !pt.ptr
        %2 = builtin.unrealized_conversion_cast %1 : !pt.ptr to i1
        %3 = pt.alloc : !pt.ptr
        llvm.cond_br %2, ^bb1, ^bb2
        ^bb1:
            pt.assign * %3 = %0 : !pt.ptr, !pt.ptr
            llvm.br ^bb3
        ^bb2:
            pt.assign * %3 = %1 : !pt.ptr, !pt.ptr
            llvm.br ^bb3
        ^bb3:
            func.return
    }
} ): () -> ()

// CHECK: mem_alloc at: loc({{.*}}:8:14) -> {mem_alloc at: loc({{.*}}:{{[5,6]}}:14), mem_alloc at: loc({{.*}}:{{[5,6]}}:14)}
