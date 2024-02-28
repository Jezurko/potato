add_mlir_dialect_library(MLIRpotato
        potato.cpp
        ops.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/potato

        DEPENDS
        MLIRPotatoIncGen
        MLIRPotatoPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
    )
add_subdirectory(analysis)
add_subdirectory(passes)
