"builtin.module"() ( {
    func.func @foo() {
        %0 = pt.alloc : !pt.ptr
        %1 = pt.alloc : !pt.ptr
        %2 = pt.address %1 : (!pt.ptr) -> !pt.ptr
        pt.assign %0 = %2 : !pt.ptr, !pt.ptr
        %4 = pt.deref %0 : (!pt.ptr) -> !pt.ptr
        %5 = pt.deref %4 : (!pt.ptr) -> !pt.ptr
        func.return
    }
} ): () -> ()
