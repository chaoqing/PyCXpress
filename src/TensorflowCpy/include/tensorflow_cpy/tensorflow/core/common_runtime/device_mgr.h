
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_MGR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_MGR_H_

#include <string>
#include <vector>

#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

class DeviceAttributes;

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

  // Returns a string listing all devices.
  virtual std::string DebugString() const = 0;

  // Returns a string of all the device mapping.
  virtual std::string DeviceMappingString() const = 0;

  // Assigns *device with pointer to Device of the given name.
  // Accepts either a full device name, or just the replica-local suffix.
  virtual Status LookupDevice(const std::string& name, Device** device);

  // Check if the current device manager contains device with the given
  // incarnation ID. Looking up by incarnation IDs because they are randomly
  // generated and not intentionally reused (unlike device pointers).
  virtual bool ContainsDevice(int64_t device_incarnation);

  virtual int NumDeviceType(const std::string& type) const = 0;

  // Returns an arbitrary CPU device if one is present, otherwise return
  // nullptr.
  virtual Device* HostCPU() const;

  TF_DISALLOW_COPY_AND_ASSIGN(DeviceMgr);
};

// Size of stale device buffer for temporary storage of removed devices.
static const size_t kStaleDeviceBufferSize = 8192;

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_DEVICE_MGR_H_
