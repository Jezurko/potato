add_mlir_dialect_library(MLIRPotatoAnalysis
    pt.cpp
    utils.cpp

    LINK_LIBS PUBLIC
    MLIRpotato
)
