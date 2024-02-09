option(NINJA_COLOUR "Enable colourful compiler output with ninja." OFF)
if(NINJA_COLOUR)
    add_compile_options(-fdiagnostics-color=always)
endif()
