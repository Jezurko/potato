module {
  llvm.func @main(%arg0: i32, %arg1: !llvm.ptr<ptr<i8>>) -> i32 {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr<i32>
    llvm.store %arg0, %1 : !llvm.ptr<i32>
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.alloca %2 x i32 : (i64) -> !llvm.ptr<i32>
    %4 = llvm.load %1 : !llvm.ptr<i32>
    llvm.store %4, %3 : !llvm.ptr<i32>
    %5 = llvm.mlir.constant(1 : index) : i64
    %6 = llvm.alloca %5 x i32 : (i64) -> !llvm.ptr<i32>
    %7 = llvm.mlir.constant(5 : i32) : i32
    llvm.store %7, %6 : !llvm.ptr<i32>
    %8 = llvm.mlir.constant(1 : index) : i64
    %9 = llvm.alloca %8 x !llvm.ptr<i32> : (i64) -> !llvm.ptr<ptr<i32>>
    %10 = llvm.load %3 : !llvm.ptr<i32>
    %11 = llvm.trunc %10 : i32 to i1
    llvm.cond_br %11, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    llvm.store %6, %9 : !llvm.ptr<ptr<i32>>
    llvm.br ^bb3
  ^bb2:  // pred: ^bb0
    llvm.store %3, %9 : !llvm.ptr<ptr<i32>>
    llvm.br ^bb3
  ^bb3:  // 2 preds: ^bb1, ^bb2
    %12 = llvm.load %9 : !llvm.ptr<ptr<i32>>
    %13 = llvm.load %12 : !llvm.ptr<i32>
    llvm.return %13 : i32
  }
}
