add_mlir_dialect_library(MLIRPotatoPasses
    ptpass.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/potato

    DEPENDS
    MLIRpotato

    LINK_LIBS PUBLIC
    MLIRpotato
    MLIRPotatoAnalysis
)
