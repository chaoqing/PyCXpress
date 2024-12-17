find_package(Protobuf CONFIG REQUIRED)

file(GLOB_RECURSE protos CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/**/*.proto")

add_library(proto-objects OBJECT ${protos})
set(PROTO_BINARY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../include")
make_directory(${PROTO_BINARY_DIR})

protobuf_generate(
  TARGET
  proto-objects
  LANGUAGE
  cpp
  PROTOC_OUT_DIR
  "${PROTO_BINARY_DIR}"
  IMPORT_DIRS
  "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow"
)

target_include_directories(proto-objects SYSTEM PUBLIC "$<BUILD_INTERFACE:${PROTO_BINARY_DIR}>")
target_link_libraries(proto-objects PUBLIC protobuf::libprotobuf)
