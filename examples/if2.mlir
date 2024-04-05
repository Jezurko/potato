"builtin.module"() ( {
    func.func @foo() {
        %0 = pt.alloc : !pt.ptr
        %1 = pt.alloc : !pt.ptr
        %2 = builtin.unrealized_conversion_cast %1 : !pt.ptr to i1
        llvm.cond_br %2, ^bb1, ^bb2
        ^bb1:
            %3 = pt.copy %1 : (!pt.ptr) -> !pt.ptr
            llvm.br ^bb2
        ^bb2:
            func.return
    }
} ): () -> ()
