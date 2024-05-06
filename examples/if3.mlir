module @"/home/jezko/src/potato/examples/if3.c" {
  llvm.func @main(%arg0: i32, %arg1: !llvm.ptr) -> i32 {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr
    llvm.store %arg0, %1 : i32, !llvm.ptr
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.alloca %2 x i32 : (i64) -> !llvm.ptr
    %4 = llvm.mlir.constant(5 : i32) : i32
    llvm.store %4, %3 : i32, !llvm.ptr
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.alloca %5 x !llvm.ptr : (i64) -> !llvm.ptr
    llvm.store %3, %6 : !llvm.ptr, !llvm.ptr
    %7 = llvm.load %1 : !llvm.ptr -> i32
    %8 = llvm.trunc %7 : i32 to i1
    llvm.cond_br %8, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.store %1, %6 : !llvm.ptr, !llvm.ptr
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    %9 = llvm.load %6 : !llvm.ptr -> !llvm.ptr
    %10 = llvm.load %9 : !llvm.ptr -> i32
    llvm.return %10 : i32
  }
}
