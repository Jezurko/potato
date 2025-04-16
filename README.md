# PoTATo
PoTATo, short for Points To Analysis Tool, is an experimental MLIR dialect based tool for points-to analysis.
It uses a novel approach based on a specialized domain-specific dialect to capture the points to effects of the analyzed program.
This allows PoTATo to focus only on the relevant information and also to perform optimizations that reduce the size of the problem.

From the users perspective, PoTATo simplifies the problem of implementing a points-to analysis to implementing a conversion pass to the points-to dialect.
This conversion is further simplified by the fact that PoTATo doesn't require from the user to convert control-flow and function call related operations.

The core algorithm is based on the MLIR Analysis framework that provides the necessary tools for computing the analysis.
This core algorithm is user-configurable, by choosing a preferred points-to lattice representation, or by providing a custom implementation of the lattice.

PoTATo has been presented as a poster on the 2024 EuroLLVM Developers' Meeting.
You can see the poster in a [trip report](https://blog.trailofbits.com/2024/06/21/eurollvm-2024-trip-report/) made by my colleagues from Trail of Bits.
The [poster](https://blog.trailofbits.com/wp-content/uploads/2024/06/image3.png) is not entirely up to date, but still might provide useful insight into the core concepts and goals.

# Showcase
TODO: update to the current status of PoTATo

# Building
We test building of PoTATo using `clang-19` and `lld`. We recommend this setup for the best experience.
The prerequisites for building our tool can be installed by the following command (for Ubuntu):
```
apt-get install build-essential cmake ninja-build libstdc++-12-dev llvm-19 libmlir-19 libmlir-19-dev mlir-19-tools libclang-19-dev lld

```
A guide for installing LLVM package on debian based distributions can be found [here](https://apt.llvm.org).

For building you need to provide the path to you local LLVM 19 install. For most Linux users it will be `/usr/lib/llvm-19`. Then run the cmake commands:
```
CC=clang
CXX=clang++
cmake --preset debug /
-DCMAKE_PREFIX_PATH=<path/to/llvm>

cmake --build --preset debug
```
Please note, that building with clang version <18 is not tested and might not work.
There is also a `release` preset available (it can be selected by replacing `debug` with `release`).

The binary `potato-opt` is located in `build/bin/`.

# Usage
At the moment there is a standalone `mlir-opt` based binary called `potato-opt`.
The relevant options of the tool at the moment are:
```
--points-to-pass                               -   Run the points-to analysis pass with the Andersen's analysis.
--steensgaard-points-to-pass                   -   Run the points-to analysis pass with the Steensgaard's analysis.
--llvm-ir-to-potato                            -   LLVM IR to PoTATo dialect converison
--canonicalize="region-simplify=disabled"      -   Canonicalize operations
```
For more options please refer to the `--help` option.

In the project root, there is an `examples` folder with mainly PoTATo IR and LLVM IR programs. In some cases there are also the source programs in `C`.

If you'd like to use the analysis in your own MLIR pass, you can look into `lib/dialect/passes/ptpass.cpp` for basic usage.
Please note, that because of how the MLIR analysis framework works, the dependencies are required for the tool to work properly.

# Including into a project
Currently, PoTATo is mainly an experimental standalone tool. A proper integration with the Tower of IRs - currently only available in [VAST](https://github.com/trailofbits/vast) - is a work in progress. Nonetheless, PoTATo uses CMake to build and as such it is possible to integrate it into your project. To include it into your CMake project you need to link the `MLIRPotato` project and provide the include paths for PoTATo (under the `include` directory).

## License

PoTATo is licensed according to the [Apache 2.0](LICENSE) license. PoTATo links against and uses Clang and LLVM APIs. Clang is also licensed under Apache 2.0, with [LLVM exceptions](https://github.com/llvm/llvm-project/blob/main/clang/LICENSE.TXT).

This research was developed with funding from the Defense Advanced Research Projects Agency (DARPA). The views, opinions and/or findings expressed are those of the author and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.

Distribution Statement A – Approved for Public Release, Distribution Unlimited
