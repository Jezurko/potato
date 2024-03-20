add_mlir_dialect_library(MLIRPotatoConversionPasses
    llvmtopotato.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/potato

    DEPENDS
    MLIRpotato
    MLIRPotatoConversionsIncGen

    LINK_LIBS PUBLIC
    MLIRpotato
)

