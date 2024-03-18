"builtin.module"() ( {
    %0 = pt.alloc : () -> (i1)
    %1 = pt.alloc : () -> (i1)
    pt.address_of %0 = addr_of %1 : (i1, i1) -> ()
} ): () -> ()
