cmake_minimum_required(VERSION 3.14...3.22)

project(PyCXpressTests LANGUAGES CXX)

# ---- Options ----

option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)

# --- Import tools ----

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CPM.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/tools.cmake)

# ---- Dependencies ----

CPMAddPackage(NAME TensorflowCpy SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../src/TensorflowCpy)
CPMAddPackage("gh:doctest/doctest@2.4.9")
CPMAddPackage("gh:TheLartians/Format.cmake@1.7.3")

# ---- Create binary ----

file(GLOB_RECURSE sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/**/*.cpp)
add_executable(${PROJECT_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${sources})
target_link_libraries(${PROJECT_NAME} TensorflowCpy doctest::doctest)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

# enable compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  target_compile_options(TensorflowCpy PUBLIC -Wall -Wpedantic -Wextra -Werror)
elseif(MSVC)
  target_compile_options(TensorflowCpy PUBLIC /W4 /WX)
  target_compile_definitions(${PROJECT_NAME} PUBLIC DOCTEST_CONFIG_USE_STD_HEADERS)
endif()

# ---- Add TensorflowCpyTests ----

enable_testing()

# Note: doctest and similar testing frameworks can automatically configure CMake tests. For other
# testing frameworks add the tests target instead: add_test(NAME ${PROJECT_NAME} COMMAND
# ${PROJECT_NAME})

include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake)
doctest_discover_tests(${PROJECT_NAME})

# ---- code coverage ----

if(ENABLE_TEST_COVERAGE)
  target_compile_options(TensorflowCpy PUBLIC -O0 -g -fprofile-arcs -ftest-coverage)
  target_link_options(TensorflowCpy PUBLIC -fprofile-arcs -ftest-coverage)
endif()
