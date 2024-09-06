add_mlir_dialect_library(MLIRPotatoPasses
    ptpass.cpp
    llvmptpass.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/

    DEPENDS
    MLIRpotato

    LINK_LIBS PUBLIC
    MLIRpotato
    MLIRPotatoAnalysis
)
