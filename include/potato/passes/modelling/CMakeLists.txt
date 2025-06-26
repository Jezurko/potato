set(POINTSTO_CONFIG_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/config.hpp
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/default-pointsto-config.yaml
    ${CMAKE_CURRENT_BINARY_DIR}/default-pointsto-config.yaml
    COPYONLY
)
