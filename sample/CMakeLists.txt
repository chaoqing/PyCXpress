cmake_minimum_required(VERSION 3.14...3.22)

# list(APPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/libtorch") list(APPEND
# CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/libtensorflow_cc")

project(Sample LANGUAGES CXX)

# --- Import tools ----

include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/CPM.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/options.cmake) # set default compiler options
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/summary.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/tools.cmake)

# ---- Dependencies ----

CPMAddPackage(NAME TensorflowCpy SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../src/TensorflowCpy)

CPMAddPackage(
  GITHUB_REPOSITORY jarro2783/cxxopts
  VERSION 3.0.0
  OPTIONS "CXXOPTS_BUILD_EXAMPLES NO" "CXXOPTS_BUILD_TESTS NO" "CXXOPTS_ENABLE_INSTALL YES"
)

# PackageProject.cmake will be used to make our target installable
# CPMAddPackage("gh:TheLartians/PackageProject.cmake@1.8.0")

CPMAddPackage(
  NAME fmt
  GIT_TAG 10.2.1
  GITHUB_REPOSITORY fmtlib/fmt
  OPTIONS "FMT_INSTALL YES" # create an installable target so no explict export not needed
)

CPMAddPackage(
  NAME magic_enum
  GIT_TAG v0.9.6
  GITHUB_REPOSITORY Neargye/magic_enum
)
# Make magic_enum exportable by adding it to the export set
install(
  TARGETS magic_enum
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include
)

# ---- Create sample executable ----

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME} ${sources})

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17 OUTPUT_NAME "sample")

target_include_directories(${PROJECT_NAME} PRIVATE ${magic_enum_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} TensorflowCpy cxxopts fmt::fmt magic_enum::magic_enum)

print_summary()
