
#ifndef TENSORFLOW_CPY_CORE_COMMON_RUNTIME_GPU_GPU_INIT_H_
#define TENSORFLOW_CPY_CORE_COMMON_RUNTIME_GPU_GPU_INIT_H_

#include <string>

#include "../../../core/platform/status.h"

// clang-format off
namespace tensorflow_cpy {
namespace stream_executor {
class Platform;
}  // namespace stream_executor
}  // namespace tensorflow_cpy
// clang-format on

// clang-format off
namespace tensorflow_cpy {
namespace tensorflow {
  using namespace ::tensorflow;

// Initializes the GPU platform and returns OK if the GPU
// platform could be initialized.
Status ValidateGPUMachineManager();

// Returns the GPU machine manager singleton, creating it and
// initializing the GPUs on the machine if needed the first time it is
// called.  Must only be called when there is a valid GPU environment
// in the process (e.g., ValidateGPUMachineManager() returns OK).
stream_executor::Platform* GPUMachineManager();

// Returns the string describing the name of the GPU platform in use.
// This value is "CUDA" by default, and
// "ROCM" when TF is built with `--config==rocm`
std::string GpuPlatformName();

}  // namespace tensorflow
}  // namespace tensorflow_cpy
// clang-format on


#endif  // TENSORFLOW_CPY_CORE_COMMON_RUNTIME_GPU_GPU_INIT_H_
