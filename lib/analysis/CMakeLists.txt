add_mlir_dialect_library(MLIRPotatoAnalysis
    pt.cpp
    fsanalysis.cpp

    LINK_LIBS PUBLIC
    MLIRpotato
)
add_subdirectory(trad)
