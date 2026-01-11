set(LLVM_TARGET_DEFINITIONS analysis.td)
mlir_tablegen(Analysis.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRPotatoPassesIncGen)

