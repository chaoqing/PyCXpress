# this script adds all subprojects to a single build to allow IDEs understand the full project
# structure.

cmake_minimum_required(VERSION 3.14...3.22)

project(BuildAll LANGUAGES CXX)

include(cmake/tools.cmake)

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# needed to generate test target
enable_testing()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/sample ${CMAKE_BINARY_DIR}/sample)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tests ${CMAKE_BINARY_DIR}/tests)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/doc ${CMAKE_BINARY_DIR}/doc)
