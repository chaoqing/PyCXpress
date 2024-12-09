
#ifndef TENSORFLOW_CORE_UTIL_STREAM_EXECUTOR_UTIL_H_
#define TENSORFLOW_CORE_UTIL_STREAM_EXECUTOR_UTIL_H_

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/stream_executor/device_memory.h"

namespace tensorflow {
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

#endif  // TENSORFLOW_CORE_UTIL_STREAM_EXECUTOR_UTIL_H_
