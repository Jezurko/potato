# PoTATo
PoTATo, short for Points To Analysis Tool, is an experimental MLIR dialect based tool for points-to analysis.
It uses a novel approach based on a specialized domain-specific dialect to capture the points to effects of the analyzed program.
This allows PoTATo to focus only on the relevant information and also to perform optimizations that reduce the size of the problem.

From the users perspective, PoTATo simplifies the problem of implementing a points-to analysis to implementing a conversion pass to the points-to dialect.
This conversion is further simplified by the fact, that PoTATo doesn't require from the user to convert control-flow and function call related operations.

The core algorithm is based on the MLIR Analysis framework, that provides the necesssary tools for computing the analysis.
This core algorithm is user-configurable, by choosing a prefered points-to lattice representation, or by providing a custom implementation of the lattice.

# Showcase
Let's take a simple LLVM IR program:
```
builtin.module {
    %one = llvm.mlir.constant(1 : index) : i64
    %a1 = llvm.alloca %one x i32 : (i64) -> !llvm.ptr<i32>
    %i = llvm.ptrtoint %a1 : !llvm.ptr<i32> to i64
    %one1 = llvm.mlir.constant(1 : index) : i64
    %off = llvm.add %i, %one1 : i64
    %a2 = llvm.inttoptr %off : i64 to !llvm.ptr<i32>
    %x = llvm.load %a2 : !llvm.ptr<i32>
}
```
This program can be converted to the PoTATo dialect by invcking the included LLVM IR conversion pass:
```
$ potato-opt --llvm-ir-to-potato <source-file>
builtin.module { 
    %one = pt.constant : i64
    %a = pt.alloc : !llvm.ptr<i32>
    %i = pt.copy %a : (!llvm.ptr<i32>) -> i64
    %off = pt.copy %i, %one : (i64, i64) -> i64
    %a2 = pt.copy %i : (i64) -> !llvm.ptr<i32>
    %x = pt.deref %a2 : (!llvm.ptr<i32>) -> i32
}
```
Now on this IR we can run the analysis (there is a demo pass `--points-to-pass` that can be used to see what the analysis result would look like) or we can run the MLIR canonicalization pass reducing the IR
```
potato-opt --llvm-ir-to-potato --canonicalize <source-file>
module {
  %0 = pt.alloc : !pt.ptr loc(#loc6)
  %1 = pt.deref %0 : (!pt.ptr) -> !pt.ptr loc(#loc5)
} loc(#loc)
#loc = loc("examples/canonicalization.mlir":1:1)
#loc1 = loc("examples/canonicalization.mlir":3:11)
#loc2 = loc("examples/canonicalization.mlir":4:10)
#loc3 = loc("examples/canonicalization.mlir":6:12)
#loc4 = loc("examples/canonicalization.mlir":7:11)
#loc5 = loc("examples/canonicalization.mlir":8:10)
#loc6 = loc(fused[#loc1, #loc2, #loc3, #loc4])
```
obtaining a shorter IR without operations that do not affect the points-to state.
We have also included the location debug information in this output to make it more apparent how the information can link back to the original IR.
Now we can invoke the analysis itself:
```
$ potato-opt --llvm-ir-to-potato --canonicalize --points-to-pass <source-file>
State in: loc(fused["examples/canonicalization.mlir":3:11, "examples/canonicalization.mlir":4:10, "examples/canonicalization.mlir":6:12, "examples/canonicalization.mlir":7:11])
  var0: %0 = pt.alloc : !pt.ptr -> {mem_loc0: <<NULL VALUE>>}
State in: loc("examples/canonicalization.mlir":8:10)
  var0: %0 = pt.alloc : !pt.ptr -> {mem_loc0: <<NULL VALUE>>}
  var1: %1 = pt.deref %0 : (!pt.ptr) -> !pt.ptr -> {}
module {
  %0 = pt.alloc : !pt.ptr
  %1 = pt.deref %0 : (!pt.ptr) -> !pt.ptr
}
```
The `--poitns-to-pass` runs the analysis and prints the lattice for every location in the reduce IR.

# Building
For building you need to provide the path to you local LLVM 18 install. For most Linux users it will be `/usr/lib/llvm-18`. Then run the cmake commands:
```
cmake --preset debug /
-DCMAKE_PREFIX_PATH=<path/to/llvm>

cmake --build --preset debug
```
There is also a `release` preset available (can be selected by replacing `debug` with `release`).

# Usage
At the moment there is a standalone, `mlir-opt` based, binary.
The relevant options of the tool at the moment are:
```
--points-to-pass    -   Run the points to analysis.
--llvm-ir-to-potato -   LLVM IR to PoTATo dialect converison
--canonicalize      -   Canonicalize operations
```
For more options please refer to the `--help` option.

In the project root, there is an `examples` folder with mainly PoTATo IR and LLVM IR programs. In some cases there are also the source programs in `C`.

If you'd like to use the analysis in your own MLIR pass, you can look into `lib/dialect/passes/ptpass.cpp` for basic usage.
Please note, that because of how the MLIR analysis framework works, the dependencies are required for the tool to work properly.

In the future, the best way to use PoTATo will be in combination with [VAST](https://github.com/trailofbits/vast).
The implementation of this integration is still under progress.

# Including into a project
PoTATo uses CMake to build. To include it into your CMake project you need to link the `MLIRPotato` project and provide the include paths for PoTATo.
```
TODO this section is under construction
```
