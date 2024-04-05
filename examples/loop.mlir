"builtin.module"() ( {
    func.func @foo() {
        %0 = pt.alloc : !pt.ptr
        %1 = pt.address %0 : (!pt.ptr) -> !pt.ptr
        llvm.br ^bb1
        ^bb1:
            %2 = pt.alloc : !pt.ptr
            %3 = pt.copy %2 : (!pt.ptr) -> !pt.ptr
            %4 = builtin.unrealized_conversion_cast %0 : !pt.ptr to i1
            llvm.cond_br %4, ^bb1, ^bb2
        ^bb2:
            func.return
    }
} ): () -> ()
