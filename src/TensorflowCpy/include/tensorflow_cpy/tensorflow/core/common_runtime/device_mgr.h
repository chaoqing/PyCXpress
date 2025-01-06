
#ifndef TENSORFLOW_CPY_CORE_COMMON_RUNTIME_DEVICE_MGR_H_
#define TENSORFLOW_CPY_CORE_COMMON_RUNTIME_DEVICE_MGR_H_

#include <tensorflow/core/framework/device_attributes.pb.h>

#include <string>
#include <vector>

#include "../../core/framework/device.h"
#include "../../core/platform/macros.h"

// clang-format off
namespace tensorflow_cpy {
namespace tensorflow {

  using namespace ::tensorflow;

// Represents a set of devices.
class DeviceMgr {
 public:
  DeviceMgr() = default;
  virtual ~DeviceMgr();

  // Returns attributes of all devices.
  virtual void ListDeviceAttributes(
      std::vector<DeviceAttributes>* devices) const = 0;

  // Returns raw pointers to the underlying devices.
  virtual std::vector<Device*> ListDevices() const = 0;

  virtual int NumDeviceType(const std::string& type) const = 0;

  // Returns an arbitrary CPU device if one is present, otherwise return
  // nullptr.
  virtual Device* HostCPU() const;

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceMgr);
};

}  // namespace tensorflow
}  // namespace tensorflow_cpy
// clang-format on


#endif  // TENSORFLOW_CPY_CORE_COMMON_RUNTIME_DEVICE_MGR_H_
