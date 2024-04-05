"builtin.module"() ( {
    %0 = pt.alloc : !pt.ptr
    %1 = pt.alloc : !pt.ptr
    %2 = pt.alloc : !pt.ptr
    pt.assign %1 = %0 : !pt.ptr, !pt.ptr
    %3 = pt.deref %1 : (!pt.ptr) -> !pt.ptr
    pt.assign %2 = %3 : !pt.ptr, !pt.ptr
} ): () -> ()
