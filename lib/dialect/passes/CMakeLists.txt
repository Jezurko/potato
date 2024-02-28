add_mlir_dialect_library(MLIRPotatoPasses
    ptpass.cpp

    LINK_LIBS PUBLIC
    MLIRpotato
    MLIRPotatoAnalysis
)
