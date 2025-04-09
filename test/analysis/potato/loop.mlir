// RUN: %potato-opt %s --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %potato-opt %s --canonicalize="region-simplify=disabled" --points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %potato-opt --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null
// RUN: %potato-opt --canonicalize="region-simplify=disabled" --steensgaard-points-to-pass="print_lattice=true print_stats=false print_func_stats=false" -o /dev/null

module {
  func.func @may_alias(%arg0: !pt.ptr, %arg1: !pt.ptr) {
    return
  }
  func.func @foo() {
    %0 = pt.alloc : !pt.ptr
    %1 = pt.alloc : !pt.ptr
    "pt.br"(%1)[^bb1] <{succ_operand_segments = array<i32: 1>}> : (!pt.ptr) -> ()
  ^bb1(%2: !pt.ptr):  // 2 preds: ^bb0, ^bb1
    %3 = pt.copy %2 : (!pt.ptr) -> !pt.ptr
    %4 = pt.alloc : !pt.ptr
    %5 = pt.copy %4 : (!pt.ptr) -> !pt.ptr
    %6 = builtin.unrealized_conversion_cast %0 : !pt.ptr to i1
    "pt.br"(%4, %3)[^bb1, ^bb2] <{succ_operand_segments = array<i32: 1, 1>}> : (!pt.ptr, !pt.ptr) -> ()
  ^bb2(%7: !pt.ptr):  // pred: ^bb1
    call @may_alias(%7, %4) : (!pt.ptr, !pt.ptr) -> ()
    return
  }
}

