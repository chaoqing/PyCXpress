
#ifndef TENSORFLOW_CPY_CORE_COMMON_RUNTIME_DEVICE_DEVICE_ID_UTILS_H_
#define TENSORFLOW_CPY_CORE_COMMON_RUNTIME_DEVICE_DEVICE_ID_UTILS_H_

#include <string>

#include "../../../core/common_runtime/device/device_id.h"
#include "../../../stream_executor/platform.h"

// clang-format off
namespace tensorflow_cpy {
namespace tensorflow {
  using namespace ::tensorflow;
namespace se = stream_executor;

// Utility methods for translation between TensorFlow device ids and platform
// device ids.
class DeviceIdUtil {
 public:
  // Convenient methods for getting the associated executor given a TfDeviceId
  // or PlatformDeviceId.
  static se::port::StatusOr<se::StreamExecutor*> ExecutorForPlatformDeviceId(
      se::Platform* device_manager, PlatformDeviceId platform_device_id) {
    return device_manager->ExecutorForDevice(platform_device_id.value());
  }
};

}  // namespace tensorflow
}  // namespace tensorflow_cpy
// clang-format on


#endif  // TENSORFLOW_CPY_CORE_COMMON_RUNTIME_DEVICE_DEVICE_ID_UTILS_H_
