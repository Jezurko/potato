"builtin.module"() ( {
    func.func @foo() {
        %0 = pt.alloc : !pt.ptr
        %1 = pt.alloc : !pt.ptr
        %2 = pt.alloc : !pt.ptr
        %3 = builtin.unrealized_conversion_cast %0 : !pt.ptr to i1
        cf.cond_br %3, ^bb1, ^bb2
        ^bb1:
            %4 = pt.copy %1 : (!pt.ptr) -> !pt.ptr
            llvm.br ^bb3
        ^bb2:
            %5 = pt.copy %1 : (!pt.ptr) -> !pt.ptr
            llvm.br ^bb3
        ^bb3:
            %6 = func.call @foo2(%0, %1) : (!pt.ptr, !pt.ptr) -> (!pt.ptr)
            func.return
    }
    func.func @foo2(!pt.ptr, !pt.ptr) -> !pt.ptr {
        ^bb0(%0: !pt.ptr, %1: !pt.ptr):
            %2 = pt.alloc : !pt.ptr
            %3 = pt.alloc : !pt.ptr
            func.return %0 : !pt.ptr
    }

} ): () -> ()
