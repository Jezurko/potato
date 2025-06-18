// RUN: %potato-opt %s --llvm-ir-to-potato -o - | %file-check %s

llvm.func @foo(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) -> !llvm.ptr attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.zero : !llvm.ptr
  %2 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %4 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %arg0, %3 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.store %arg1, %4 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  %5 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  %6 = llvm.icmp "ne" %5, %1 : !llvm.ptr
  llvm.cond_br %6, ^bb2, ^bb1
^bb1:  // pred: ^bb0
  %7 = llvm.load %4 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  llvm.store %7, %2 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.br ^bb3
^bb2:  // pred: ^bb0
  %8 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  llvm.store %8, %2 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.br ^bb3
^bb3:  // 2 preds: ^bb1, ^bb2
  %9 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  llvm.return %9 : !llvm.ptr
}
llvm.func @main() -> i32 attributes {dso_local, frame_pointer = #llvm.framePointerKind<all>, no_inline, no_unwind, optimize_none, passthrough = [["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>, tune_cpu = "generic", uwtable_kind = #llvm.uwtableKind<async>} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: [[FOO_ADDR:%[0-9]+]] = pt.address @foo
  %2 = llvm.mlir.addressof @foo : !llvm.ptr
  %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %6 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %7 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  llvm.store %1, %3 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.store %1, %4 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.store %1, %5 {alignment = 4 : i64} : i32, !llvm.ptr
  // CHECK: pt.assign * [[FOO_ALLOCA:%[0-9]+]] = [[FOO_ADDR]]
  llvm.store %2, %7 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  // CHECK: [[FOO_LOAD:%[0-9]+]] = pt.deref [[FOO_ALLOCA]]
  %8 = llvm.load %7 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  // CHECK: pt.call_indirect [[FOO_LOAD]](
  %9 = llvm.call %8(%4, %5) : !llvm.ptr, (!llvm.ptr {llvm.noundef}, !llvm.ptr {llvm.noundef}) -> !llvm.ptr
  llvm.store %9, %6 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  %10 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
  llvm.return %10 : i32
}
