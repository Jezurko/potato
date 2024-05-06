"builtin.module"() ( {
    func.func @fun() {
        %0 = pt.alloc : !pt.ptr
        %1 = pt.address %0 : (!pt.ptr) -> !pt.ptr
        func.return
    }
} ): () -> ()
