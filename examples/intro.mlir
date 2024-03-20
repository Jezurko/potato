"builtin.module"() ( {
    %0 = pt.alloc : i1
    %1 = pt.address_of %0 : (i1) -> (i1)
} ): () -> ()
