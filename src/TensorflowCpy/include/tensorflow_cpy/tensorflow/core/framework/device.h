
// A Device is a something that can perform computations as part of a
// model.  Devices can be local (runs computation on this machine), or
// remote (contacts a device local to another machine using an RPC to
// do the work).  Devices are registered in a DeviceSet, which is also
// responsible for the Device <-> id mapping.
//
// Device names
// * Every Device should have a unique name with the format:
//     /job:___/replica:___/task:___/(gpu|cpu):___
//   An example name would be "/job:train/replica:0/task:3/device:GPU:2".
// * Task numbers are within the specified replica, so there are as
//   many "task zeros" as replicas.

#ifndef TENSORFLOW_CORE_FRAMEWORK_DEVICE_H_
#define TENSORFLOW_CORE_FRAMEWORK_DEVICE_H_

#include <string>

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
class DeviceBase {
 public:
  explicit DeviceBase(Env* env) : env_(env) {}
  virtual ~DeviceBase();

  Env* env() const { return env_; }


  // Return the Allocator implementation to use based on the allocator
  // attributes requested.  See allocator.h for more details.
  virtual Allocator* GetAllocator(AllocatorAttributes /*attr*/) {
    LOG(FATAL) << "GetAllocator() is not implemented.";
    return nullptr;
  }

  // Unimplemented by default
  virtual const DeviceAttributes& attributes() const;
  virtual const std::string& name() const;


 private:
  Env* const env_;
};

class Device : public DeviceBase {
 public:
  // Callback type that takes a Status and returns void.
  typedef std::function<void(const Status&)> DoneCallback;

  Device(Env* env, const DeviceAttributes& device_attributes);
  ~Device() override;

  // Full name of this device (see top comment).
  const std::string& name() const override { return device_attributes_.name(); }


  // Describes what kind of device this is.  This is intended to be
  // human-readable and not computer-parsed, except that two devices
  // with the same device_type() are expected to perform similarly
  // (both from a computation and communication perspective).
  const std::string& device_type() const {
    return device_attributes_.device_type();
  }

  // Returns an aggregation of device attributes.
  const DeviceAttributes& attributes() const override {
    return device_attributes_;
  }

  // Summarizes the status of this Device, for debugging.
  std::string DebugString() const { return device_attributes_.DebugString(); }


  virtual bool IsLocal() const { return true; }

 protected:
 private:
  DeviceAttributes device_attributes_;
  TF_DISALLOW_COPY_AND_ASSIGN(Device);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_DEVICE_H_
