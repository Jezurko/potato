"builtin.module"() ( {
    func.func @foo() {
        %0 = pt.alloc : i1
        %1 = pt.address_of %0 : (i1) -> (i1)
        llvm.br ^bb1
        ^bb1:
            %2 = pt.alloc : i1
            pt.copy %1 = %2 : i1, i1
            llvm.cond_br %0, ^bb1, ^bb2
        ^bb2:
            func.return
    }
} ): () -> ()
