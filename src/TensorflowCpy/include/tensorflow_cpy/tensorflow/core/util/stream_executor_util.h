
#ifndef TENSORFLOW_CPY_CORE_UTIL_STREAM_EXECUTOR_UTIL_H_
#define TENSORFLOW_CPY_CORE_UTIL_STREAM_EXECUTOR_UTIL_H_

#include "../../core/framework/tensor.h"
#include "../../stream_executor/device_memory.h"

// clang-format off
namespace tensorflow_cpy {
namespace tensorflow {
  using namespace ::tensorflow;
  namespace se = stream_executor;

// StreamExecutorUtil contains functions useful for interfacing
// between StreamExecutor classes and TensorFlow.
class StreamExecutorUtil {
 public:
  // Map a Tensor as a DeviceMemory object wrapping the given typed
  // buffer.
  template <typename T>
  static se::DeviceMemory<T> AsDeviceMemory(const Tensor& t) {
    T* ptr = reinterpret_cast<T*>(const_cast<void*>(t.data()));
    return se::DeviceMemory<T>(se::DeviceMemoryBase(ptr, t.TotalBytes()));
  }
};

}  // namespace tensorflow
}  // namespace tensorflow_cpy
// clang-format on


#endif  // TENSORFLOW_CPY_CORE_UTIL_STREAM_EXECUTOR_UTIL_H_
