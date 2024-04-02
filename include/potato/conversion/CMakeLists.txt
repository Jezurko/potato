set(LLVM_TARGET_DEFINITIONS conversions.td)
add_subdirectory(type)
mlir_tablegen(Conversions.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRPotatoConversionsIncGen)
