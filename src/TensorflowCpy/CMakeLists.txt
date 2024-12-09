cmake_minimum_required(VERSION 3.14...3.22)

# ---- Project ----

# Note: update this to your new project's name and version
project(TensorflowCpy LANGUAGES CXX)

set(PROJECT_SNAKE_NAME tensorflow_cpy)

# ---- Include guards ----

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

if(ENABLE_TORCH)
  find_package(Torch REQUIRED)
else()
  if(USE_LIBTENSORFLOW_CC)
    find_package(TensorFlow REQUIRED)
  else()
    #find_package(Protobuf REQUIRED)
  endif()
endif()

# ---- Add source files ----

# Note: globbing sources is considered bad practice as CMake's generators may not detect new files
# automatically. Keep that in mind when changing files, or explicitly mention them here.
file(GLOB_RECURSE headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/**/*.h")
file(GLOB_RECURSE sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp")

# ---- Create library ----
# Note: for header-only libraries change all PUBLIC flags to INTERFACE and create an interface
# target: add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME} ${headers} ${sources})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

# being a cross-platform target, we enforce standards conformance on MSVC
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")

# include options
target_include_directories(${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
if(ENABLE_TORCH)
  target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
else()
  if(USE_LIBTENSORFLOW_CC)
    target_include_directories(${PROJECT_NAME} PRIVATE ${TensorFlow_INCLUDE_DIRS})
  else()
    target_include_directories(
      ${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include/tensorflow_cpy>
    )
    target_include_directories(
      ${PROJECT_NAME} PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/../../third_party/protobuf/include>
    )
    #target_include_directories(${PROJECT_NAME} PRIVATE ${Protobuf_INCLUDE_DIRS})
  endif()
endif()

# link options
if(ENABLE_TORCH)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})
else()
  if(USE_LIBTENSORFLOW_CC)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TensorFlow_LIBRARIES})
  else()
    #target_link_libraries(${PROJECT_NAME} PRIVATE ${Protobuf_LIBRARIES})
  endif()
endif()
