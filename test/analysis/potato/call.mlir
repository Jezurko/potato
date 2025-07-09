// RUN: %potato-opt %s --llvm-ir-to-potato --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null | %file-check %s
llvm.func @foo2(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
  llvm.call @may_alias(%arg0, %arg1) : (!llvm.ptr, !llvm.ptr) -> ()
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(3 : i32) : i32
  %2 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
  %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.store %arg0, %2 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.store %arg1, %3 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
  llvm.store %1, %4 {alignment = 4 : i64} : i32, !llvm.ptr
  %5 = llvm.load %4 {alignment = 4 : i64} : !llvm.ptr -> i32
  %6 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
  llvm.store %5, %6 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.return
}
llvm.func @foo() attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
  %0 = llvm.mlir.constant(1 : i32) : i32
  %1 = llvm.mlir.constant(2 : i32) : i32
  %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
  llvm.store %0, %2 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.store %1, %3 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.call @foo2(%2, %3) : (!llvm.ptr, !llvm.ptr) -> ()
  llvm.call @foo2(%3, %2) : (!llvm.ptr, !llvm.ptr) -> ()
  %5 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
  llvm.store %5, %4 {alignment = 4 : i64} : i32, !llvm.ptr
  llvm.return
}
llvm.func @may_alias(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
  llvm.return
}


// CHECK-DAG: <block argument> of type '!pt.ptr' at index: 0 -> {mem_alloc at: loc({{.*}}), mem_alloc at: loc({{.*}})}
// CHECK-DAG: <block argument> of type '!pt.ptr' at index: 1 -> {mem_alloc at: loc({{.*}}), mem_alloc at: loc({{.*}})}
