cmake_minimum_required(VERSION 3.14...3.22)

project(Example LANGUAGES CXX)

# ---- Dependencies ----
if(NOT TARGET PyCXpress)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../ EXTERNAL)
endif()

# ---- Create sample executable ----

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(${PROJECT_NAME} ${sources})

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17 OUTPUT_NAME "example")

target_link_libraries(${PROJECT_NAME} PyCXpress)
