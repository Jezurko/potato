module {
  llvm.func @main() -> i32 {
    %0 = llvm.mlir.constant(1 : index) : i64
    %1 = llvm.alloca %0 x i32 : (i64) -> !llvm.ptr<i32>
    %2 = llvm.mlir.constant(5 : i32) : i32
    llvm.store %2, %1 : !llvm.ptr<i32>
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.alloca %3 x i32 : (i64) -> !llvm.ptr<i32>
    %5 = llvm.mlir.constant(6 : i32) : i32
    llvm.store %5, %4 : !llvm.ptr<i32>
    %6 = llvm.mlir.constant(1 : index) : i64
    %7 = llvm.alloca %6 x !llvm.ptr<i32> : (i64) -> !llvm.ptr<ptr<i32>>
    llvm.store %1, %7 : !llvm.ptr<ptr<i32>>
    %8 = llvm.load %7 : !llvm.ptr<ptr<i32>>
    %9 = llvm.mlir.constant(0 : i32) : i32
    %10 = llvm.mlir.null : !llvm.ptr<i8>
    %11 = llvm.bitcast %10 : !llvm.ptr<i8> to !llvm.ptr<i32>
    %12 = llvm.icmp "ne" %8, %11 : !llvm.ptr<i32>
    %13 = llvm.zext %12 : i1 to i32
    %14 = llvm.trunc %13 : i32 to i1
    llvm.cond_br %14, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    %15 = llvm.load %7 : !llvm.ptr<ptr<i32>>
    %16 = llvm.load %4 : !llvm.ptr<i32>
    llvm.store %16, %15 : !llvm.ptr<i32>
    llvm.br ^bb2
  ^bb2:  // 2 preds: ^bb0, ^bb1
    %17 = llvm.load %1 : !llvm.ptr<i32>
    llvm.return %17 : i32
  }
}
