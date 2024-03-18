"builtin.module"() ( {
    func.func @foo() {
        %0 = pt.alloc : () -> (i1)
        %1 = pt.alloc : () -> (i1)
        %2 = pt.alloc : () -> (i1)
        llvm.cond_br %0, ^bb1, ^bb2
        ^bb1:
            pt.address_of %0 = addr_of %1 : (i1, i1) -> ()
            llvm.br ^bb3
        ^bb2:
            pt.address_of %0 = addr_of %2 : (i1, i1) -> ()
            llvm.br ^bb3
        ^bb3:
            %4 = func.call @foo2(%0) : (i1) -> (i1)
            func.return
    }
    func.func private @foo2(i1) -> i1 {
        ^bb0(%0: i1):
            %1 = pt.alloc : () -> (i1)
            pt.assign *%0 = %1 : (i1, i1) -> ()
            func.return %0 : i1
    }

} ): () -> ()
