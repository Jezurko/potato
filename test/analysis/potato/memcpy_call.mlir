// RUN: %potato-opt %s --llvm-ir-to-potato --add-fn-bodies-from-models --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %potato-opt %s --llvm-ir-to-potato --add-fn-bodies-from-models --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %potato-opt %s --llvm-ir-to-potato --add-fn-bodies-from-models --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %potato-opt %s --llvm-ir-to-potato --add-fn-bodies-from-models --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %potato-opt %s --llvm-ir-to-potato --add-fn-bodies-from-models="inline-model-bodies=false" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %potato-opt %s --llvm-ir-to-potato --add-fn-bodies-from-models="inline-model-bodies=false" --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %potato-opt %s --llvm-ir-to-potato --add-fn-bodies-from-models="inline-model-bodies=false" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %potato-opt %s --llvm-ir-to-potato --add-fn-bodies-from-models="inline-model-bodies=false" --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>} {
  llvm.func @must_alias(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %1 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg1, %2 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.return
  }
  llvm.func @may_alias(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %1 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg1, %2 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.return
  }
  llvm.func @no_alias(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}) attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %0 x !llvm.ptr {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %1 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.store %arg1, %2 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.return
  }
  llvm.func @memcpy(%arg0: !llvm.ptr {llvm.noundef}, %arg1: !llvm.ptr {llvm.noundef}, %arg2: i64)
  llvm.func @main() -> i32 attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = ["noinline", "nounwind", "optnone", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(10 : i64) : i64
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.mlir.constant(24 : i64) : i64
    %4 = llvm.alloca %0 x !llvm.struct<"struct.myStruct", (f32, struct<"struct.innerStruct", (i32, ptr)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.alloca %0 x !llvm.struct<"struct.myStruct", (f32, struct<"struct.innerStruct", (i32, ptr)>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %6 = llvm.call @malloc(%1) : (i64) -> !llvm.ptr
    %7 = llvm.getelementptr inbounds %4[%2, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.myStruct", (f32, struct<"struct.innerStruct", (i32, ptr)>)>
    %8 = llvm.getelementptr inbounds %7[%2, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.innerStruct", (i32, ptr)>
    llvm.store %6, %8 {alignment = 8 : i64} : !llvm.ptr, !llvm.ptr
    llvm.call @memcpy(%5, %4, %3) : (!llvm.ptr, !llvm.ptr, i64) -> ()
    %9 = llvm.getelementptr inbounds %4[%2, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.myStruct", (f32, struct<"struct.innerStruct", (i32, ptr)>)>
    %10 = llvm.getelementptr inbounds %9[%2, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.innerStruct", (i32, ptr)>
    %11 = llvm.load %10 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    %12 = llvm.getelementptr inbounds %5[%2, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.myStruct", (f32, struct<"struct.innerStruct", (i32, ptr)>)>
    %13 = llvm.getelementptr inbounds %12[%2, 1] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<"struct.innerStruct", (i32, ptr)>
    %14 = llvm.load %13 {alignment = 8 : i64} : !llvm.ptr -> !llvm.ptr
    llvm.call @may_alias(%11, %14) : (!llvm.ptr, !llvm.ptr) -> ()
    llvm.return %2 : i32
  }
  llvm.func @malloc(i64 {llvm.noundef}) -> !llvm.ptr attributes {frame_pointer = #llvm.framePointerKind<all>, passthrough = [["allocsize", "4294967295"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["tune-cpu", "generic"]], target_cpu = "x86-64", target_features = #llvm.target_features<["+cmov", "+cx8", "+fxsr", "+mmx", "+sse", "+sse2", "+x87"]>}
}
