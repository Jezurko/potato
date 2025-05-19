// RUN: %potato-opt %s --llvm-ir-to-potato -o - | %file-check %s

"builtin.module"() ( {
    func.func @may_alias(%p : !pt.ptr, %q : !pt.ptr) {
        func.return
    }
    func.func @foo() {
        // CHECK: [[A1:%[0-9]+]] = pt.alloc
        %0 = pt.alloc : !pt.ptr
        // CHECK: [[C1:%[0-9]+]] = pt.copy [[A1]]
        %a = pt.copy %0: (!pt.ptr) -> !pt.ptr
        // CHECK: [[A2:%[0-9]+]] = pt.alloc
        %1 = pt.alloc : !pt.ptr
        // CHECK: [[C2:%[0-9]+]] = pt.copy [[A2]]
        %b = pt.copy %1 : (!pt.ptr) -> !pt.ptr
        %2 = builtin.unrealized_conversion_cast %1 : !pt.ptr to i1
        %3 = pt.alloc : !pt.ptr
        // CHECK: pt.br ^bb1([[A1]], [[C1]] : !pt.ptr, !pt.ptr), ^bb2([[A2]], [[C2]] : !pt.ptr, !pt.ptr)
        cf.cond_br %2, ^bb1(%0, %a : !pt.ptr, !pt.ptr), ^bb2(%1, %b : !pt.ptr, !pt.ptr)
        ^bb1(%arg0: !pt.ptr, %arg0b : !pt.ptr):
            func.call @may_alias(%arg0, %arg0b) : (!pt.ptr, !pt.ptr) -> ()
            func.return
        ^bb2(%arg1 : !pt.ptr, %arg1b : !pt.ptr):
            func.call @may_alias(%arg1, %arg1b) : (!pt.ptr, !pt.ptr) -> ()
            func.return
    }
} ): () -> ()
