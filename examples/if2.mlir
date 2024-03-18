"builtin.module"() ( {
    func.func @foo() {
        %0 = pt.alloc : () -> (i1)
        %1 = pt.alloc : () -> (i1)
        llvm.cond_br %0, ^bb1, ^bb2
        ^bb1:
            pt.copy %0 = %1 : (i1, i1) -> ()
            llvm.br ^bb2
        ^bb2:
            func.return
    }
} ): () -> ()
