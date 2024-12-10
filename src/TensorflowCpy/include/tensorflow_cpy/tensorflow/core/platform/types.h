
#ifndef TENSORFLOW_CPY_CORE_PLATFORM_TYPES_H_
#define TENSORFLOW_CPY_CORE_PLATFORM_TYPES_H_

#include <cstdint>

// clang-format off
namespace tensorflow_cpy {
namespace tensorflow {

typedef signed char int8;
typedef short int16;
typedef int int32;
typedef ::std::int64_t int64;

typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef std::uint64_t uint64;

}  // namespace tensorflow


namespace tensorflow {

static const uint8 kuint8max = static_cast<uint8>(0xFF);
static const uint16 kuint16max = static_cast<uint16>(0xFFFF);
static const uint32 kuint32max = static_cast<uint32>(0xFFFFFFFF);
static const uint64 kuint64max = static_cast<uint64>(0xFFFFFFFFFFFFFFFFull);
static const int8_t kint8min = static_cast<int8>(~0x7F);
static const int8_t kint8max = static_cast<int8>(0x7F);
static const int16_t kint16min = static_cast<int16>(~0x7FFF);
static const int16_t kint16max = static_cast<int16>(0x7FFF);
static const int32_t kint32min = static_cast<int32>(~0x7FFFFFFF);
static const int32_t kint32max = static_cast<int32>(0x7FFFFFFF);
static const int64_t kint64min = static_cast<int64_t>(~0x7FFFFFFFFFFFFFFFll);
static const int64_t kint64max = static_cast<int64_t>(0x7FFFFFFFFFFFFFFFll);

// A typedef for a uint64 used as a short fingerprint.
typedef uint64 Fprint;

}  // namespace tensorflow
}  // namespace tensorflow_cpy
// clang-format on


#endif  // TENSORFLOW_CPY_CORE_PLATFORM_TYPES_H_
