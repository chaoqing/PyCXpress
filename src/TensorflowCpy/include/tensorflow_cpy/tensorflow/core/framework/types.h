
#ifndef TENSORFLOW_CPY_CORE_FRAMEWORK_TYPES_H_
#define TENSORFLOW_CPY_CORE_FRAMEWORK_TYPES_H_

#include <tensorflow/core/framework/types.pb.h>

#include <string>

#include "../../core/platform/macros.h"
#include "../../core/platform/types.h"

// clang-format off
namespace tensorflow_cpy {
namespace tensorflow {
  using namespace ::tensorflow;

// MemoryType is used to describe whether input or output Tensors of
// an OpKernel should reside in "Host memory" (e.g., CPU memory) or
// "Device" Memory (CPU memory for CPU devices, GPU memory for GPU
// devices).
enum MemoryType {
  DEVICE_MEMORY = 0,
  HOST_MEMORY = 1,
};

// A DeviceType is just a string, but we wrap it up in a class to give
// some type checking as we're passing these around
class DeviceType {
 public:
  DeviceType(const char* type)  // NOLINT(runtime/explicit)
      : type_(type) {}

  const char* type() const { return type_.c_str(); }
  const std::string& type_string() const { return type_; }

 private:
  std::string type_;
};

// Convenient constants that can be passed to a DeviceType constructor
TF_EXPORT extern const char* const DEVICE_DEFAULT;     // "DEFAULT"
TF_EXPORT extern const char* const DEVICE_CPU;         // "CPU"
TF_EXPORT extern const char* const DEVICE_GPU;         // "GPU"


// DataTypeToEnum<T>::v() and DataTypeToEnum<T>::value are the DataType
// constants for T, e.g. DataTypeToEnum<float>::v() is DT_FLOAT.
template <class T>
struct DataTypeToEnum {
};  // Specializations below

// EnumToDataType<VALUE>::Type is the type for DataType constant VALUE, e.g.
// EnumToDataType<DT_FLOAT>::Type is float.
template <DataType VALUE>
struct EnumToDataType {};  // Specializations below

// Template specialization for both DataTypeToEnum and EnumToDataType.
#define MATCH_TYPE_AND_ENUM(TYPE, ENUM)                 \
  template <>                                           \
  struct DataTypeToEnum<TYPE> {                         \
    static DataType v() { return ENUM; }                \
    static constexpr DataType value = ENUM;             \
  };                                                    \
  template <>                                           \
  struct EnumToDataType<ENUM> {                         \
    typedef TYPE Type;                                  \
  }

MATCH_TYPE_AND_ENUM(float, DT_FLOAT);
MATCH_TYPE_AND_ENUM(double, DT_DOUBLE);
MATCH_TYPE_AND_ENUM(int32, DT_INT32);
MATCH_TYPE_AND_ENUM(uint32, DT_UINT32);
MATCH_TYPE_AND_ENUM(uint16, DT_UINT16);
MATCH_TYPE_AND_ENUM(uint8, DT_UINT8);
MATCH_TYPE_AND_ENUM(int16, DT_INT16);
MATCH_TYPE_AND_ENUM(int8, DT_INT8);
MATCH_TYPE_AND_ENUM(bool, DT_BOOL);

template <>
struct DataTypeToEnum<long> {
  static DataType v() { return value; }
  static constexpr DataType value = sizeof(long) == 4 ? DT_INT32 : DT_INT64;
};
template <>
struct EnumToDataType<DT_INT64> {
  typedef int64_t Type;
};

template <>
struct DataTypeToEnum<unsigned long> {
  static DataType v() { return value; }
  static constexpr DataType value =
      sizeof(unsigned long) == 4 ? DT_UINT32 : DT_UINT64;
};
template <>
struct EnumToDataType<DT_UINT64> {
  typedef tensorflow::uint64 Type;
};

template <>
struct DataTypeToEnum<long long> {
  static DataType v() { return DT_INT64; }
  static constexpr DataType value = DT_INT64;
};

template <>
struct DataTypeToEnum<unsigned long long> {
  static DataType v() { return DT_UINT64; }
  static constexpr DataType value = DT_UINT64;
};
#undef MATCH_TYPE_AND_ENUM


// Returns a 0 on failure
int DataTypeSize(DataType dt);

}  // namespace tensorflow
}  // namespace tensorflow_cpy
// clang-format on


#endif  // TENSORFLOW_CPY_CORE_FRAMEWORK_TYPES_H_
