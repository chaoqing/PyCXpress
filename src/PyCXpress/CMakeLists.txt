cmake_minimum_required(VERSION 3.14...3.22)

# ---- Project ----

project(PyCXpress LANGUAGES CXX)

# ---- Include guards ----

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

# ---- Add source files ----

# Note: globbing sources is considered bad practice as CMake's generators may not detect new files
# automatically. Keep that in mind when changing files, or explicitly mention them here.
file(GLOB_RECURSE headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/**/*.hpp")

# ---- Create library ----
# Note: for header-only libraries change all PUBLIC flags to INTERFACE and create an interface
# target: add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME} INTERFACE ${headers})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

# being a cross-platform target, we enforce standards conformance on MSVC
target_compile_options(${PROJECT_NAME} INTERFACE "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")

# compile with python3 shared library
execute_process(
  COMMAND python3-config --cflags
  OUTPUT_VARIABLE PYTHON3_CFLAGS
  RESULT_VARIABLE PYTHON3_CONFIG_STATUS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT PYTHON3_CONFIG_STATUS EQUAL 0)
  message(
    FATAL_ERROR
      "python3-config does not exist, considering install it with 'apt install python3-dev' or build python with '--enable-shared'"
  )
endif()
target_compile_options(${PROJECT_NAME} INTERFACE "SHELL:${PYTHON3_CFLAGS}")

# link python3 shared library
execute_process(
  COMMAND python3-config --ldflags --embed
  OUTPUT_VARIABLE PYTHON3_EMBED_FLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
# do not use target_link_options because cmake add those options before objects
target_link_libraries(${PROJECT_NAME} INTERFACE ${PYTHON3_EMBED_FLAGS})

# include pycxpress

# set complilation flags with cache method
set(PYTHON3_EXE
    "python3"
    CACHE STRING "Used by command line tool."
)
set(PYTHON3_EXE_INTERNAL
    ""
    CACHE INTERNAL "for internal use only; do not modify."
)

if(NOT PYTHON3_EXE STREQUAL PYTHON3_EXE_INTERNAL)
  execute_process(
    COMMAND ${PYTHON3_EXE} -c "import PyCXpress.__main__ as p; print(';'.join(p.all_includes()))"
    OUTPUT_VARIABLE PYCXPRESS_INCLUDE_FLAGS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(PYTHON3_EXE_INTERNAL
      ${PYTHON3_EXE}
      CACHE INTERNAL "for internal use only; do not modify."
  )
endif()
target_include_directories(${PROJECT_NAME} INTERFACE ${PYCXPRESS_INCLUDE_FLAGS})
