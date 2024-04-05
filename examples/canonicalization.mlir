builtin.module {
    %one = llvm.mlir.constant(1 : index) : i64
    %a1 = llvm.alloca %one x i32 : (i64) -> !llvm.ptr<i32>
    %i = llvm.ptrtoint %a1 : !llvm.ptr<i32> to i64
    %one1 = llvm.mlir.constant(1 : index) : i64
    %off = llvm.add %i, %one1 : i64
    %a2 = llvm.inttoptr %off : i64 to !llvm.ptr<i32>
    %x = llvm.load %a2 : !llvm.ptr<i32>
}
