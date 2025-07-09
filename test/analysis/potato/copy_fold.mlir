// RUN: %potato-opt %s -o - --canonicalize="region-simplify=disabled" | %file-check %s
func.func private @user(!pt.ptr)
func.func @foo(%arg0 : !pt.ptr) {
    %0 = pt.alloc : !pt.ptr
    // CHECK: pt.copy %0, %arg0
    %1 = pt.copy %0, %arg0 : (!pt.ptr, !pt.ptr) -> !pt.ptr
    call @user(%1) : (!pt.ptr) -> ()
    // CHECK: pt.copy %arg0, %0
    %2 = pt.copy %arg0, %0 : (!pt.ptr, !pt.ptr) -> !pt.ptr
    call @user(%2) : (!pt.ptr) -> ()
    %3 = pt.copy %arg0 : (!pt.ptr) -> !pt.ptr
    // CHECK: call @user(%arg0)
    call @user(%3) : (!pt.ptr) -> ()

    func.return
}
