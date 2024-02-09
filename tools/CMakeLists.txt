get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        ${dialect_libs}
        ${conversion_libs}

        MLIROptLib
        MLIRpotato
        )
add_llvm_executable(potato-opt potato-opt.cpp)

llvm_update_compile_flags(potato-opt)
target_link_libraries(potato-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(potato-opt)
