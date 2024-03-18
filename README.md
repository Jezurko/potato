# PoTATo
PoTATo, short for Points To Analysis Tool, is an experimental MLIR dialect based tool for points-to analysis.
```
TODO motivation & features
```

# Building
For building you need to provide the path to you local LLVM 17 install. For most Linux users it will be `/usr/lib/llvm-17`. Then run the cmake commands:
```
cmake --preset debug /
-DCMAKE_PREFIX_PATH=<path/to/llvm>

cmake --build --preset debug
```
There is also a `release` preset available (can be selected by replacing `debug` with `release`).

# Usage
At the moment there is a standalone binary with an example pass that prints the analysis result:

```
./build/bin/potato-opt --points-to-pass <source_file>
```

You can try one of the files from `examples` folder.

If you'd like to use the analysis in your own MLIR pass, you can look into `lib/dialect/passes/ptpass.cpp` for basic usage.

In the future, the best way to use PoTATo will be in combination with [VAST](https://github.com/trailofbits/vast).

# Including into a project
```
TODO
```
