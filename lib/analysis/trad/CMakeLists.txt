add_mlir_dialect_library(MLIRPotatoLLVMAnalysis
    llvm_aa.cpp

    LINK_LIBS PUBLIC
    MLIRIR
)
