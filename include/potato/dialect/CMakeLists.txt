add_mlir_dialect(Potato pt)

set(LLVM_TARGET_DEFINITIONS passes.td)
mlir_tablegen(Passes.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRPotatoPassesIncGen)
