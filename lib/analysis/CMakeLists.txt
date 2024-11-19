add_mlir_dialect_library(MLIRPotatoAnalysis
    function_models.cpp
    pt.cpp
    andersen.cpp
    steensgaard.cpp

    LINK_LIBS PUBLIC
    MLIRpotato
)
add_subdirectory(trad)
