
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

  // Parse `visible_device_list` into a list of platform Device ids.
  static Status ParseVisibleDeviceList(
      const std::string& visible_device_list, const int visible_device_count,
      std::vector<PlatformDeviceId>* visible_device_order) ;
};

}  // namespace tensorflow
}  // namespace tensorflow_cpy
// clang-format on


#endif  // TENSORFLOW_CPY_CORE_COMMON_RUNTIME_DEVICE_DEVICE_ID_UTILS_H_
