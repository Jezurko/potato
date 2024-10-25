include(CTest)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(POTATO_TEST_DEPENDS
  potato-opt
)

add_lit_testsuite(check-potato "Running the PoTATo regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${VAST_TEST_DEPENDS}
)

set_target_properties(check-potato PROPERTIES FOLDER "Tests")

add_lit_testsuites(POTATO ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${POTATO_TEST_DEPENDS})
add_test(NAME lit
         COMMAND lit -v "${CMAKE_CURRENT_BINARY_DIR}"
         --param BUILD_TYPE=$<CONFIG>)
