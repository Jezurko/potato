// RUN: %potato-opt %s -o %t && %potato-opt %t -o - | diff -B %t -

func.func @foo() {
    %0 = pt.alloc : !pt.ptr
    pt.br ^bb1()
    ^bb1:
        %1 = pt.alloc : !pt.ptr
        %2 = pt.copy %1 : (!pt.ptr) -> !pt.ptr
        pt.br ^bb1, ^bb2(%1, %2 : !pt.ptr, !pt.ptr)
    ^bb2(%arg0 : !pt.ptr, %arg1 : !pt.ptr):
        pt.br ^bb3(%arg0 : !pt.ptr)
    ^bb3(%arg2 : !pt.ptr):
        func.return
}
