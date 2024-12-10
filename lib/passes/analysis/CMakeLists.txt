add_mlir_dialect_library(MLIRPotatoPasses
    aabench.cpp
    ptpass.cpp
    llvmptpass.cpp
    steensgaardptpass.cpp
    steensbench.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/

    DEPENDS
    MLIRpotato

    LINK_LIBS PUBLIC
    MLIRpotato
    MLIRPotatoAnalysis
)
