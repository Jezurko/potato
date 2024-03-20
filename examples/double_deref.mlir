"builtin.module"() ( {
    func.func @foo() {
        %0 = pt.alloc : () -> (i1)
        %1 = pt.alloc : () -> (i1)
        %2 = pt.address_of %1 : (i1) -> (i1)
        %3 = pt.address_of %0 : (i1) -> (i1)
        %4 = pt.deref *%3 : (i1) -> (i1)
        func.return
    }
} ): () -> ()
