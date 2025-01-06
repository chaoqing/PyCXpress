// Defines types and declares functions for identifying and extracting
// information about the types of platforms and supporting libraries for which
// StreamExecutor implementations exist.
#ifndef TENSORFLOW_CPY_STREAM_EXECUTOR_PLATFORM_H_
#define TENSORFLOW_CPY_STREAM_EXECUTOR_PLATFORM_H_

#include <map>

#include "../core/platform/statusor.h"

// clang-format off
namespace tensorflow_cpy {
namespace stream_executor {
namespace port {
  using ::tensorflow_cpy::tensorflow::Status;
  using ::tensorflow_cpy::tensorflow::StatusOr;
};

class StreamExecutor;

// Data that describes the execution target of the StreamExecutor, in terms of
// important logical parameters. These include dimensionality limits and
// physical parameters of interest, such as number of cores present on the
// device.
//
// Thread-safe: immutable post-initialization.
class DeviceDescription {
 public:
  // Returns the PCI bus identifier for this device, of the form
  // [domain]:[bus]:[device].[function]
  const std::string &pci_bus_id() const { return pci_bus_id_; }

 private:
  DeviceDescription();

  friend StreamExecutor;

  std::string pci_bus_id_;

  SE_DISALLOW_COPY_AND_ASSIGN(DeviceDescription);
};
};

namespace stream_executor {

class StreamExecutor;
class DeviceDescription;

// Describes the platform for a StreamExecutor instantiation to act upon.
//
// Implementors: if you add a value here be sure to update PlatformKindString
// and CheckPlatformKindIsValid.
enum class PlatformKind {
  kInvalid,
  kCuda,
  kROCm,
  kOpenCL,
  kHost,
  kMock,
  kSize,
};

// Abstract base class for a platform registered with the MultiPlatformManager.
class Platform {
 public:
  virtual ~Platform();

  // Returns a device with the given ordinal on this platform with a default
  // plugin configuration or, if none can be found with the given ordinal or
  // there is an error in opening a context to communicate with the device, an
  // error status is returned.
  //
  // Ownership of the executor is NOT transferred to the caller --
  // the Platform owns the executors in a singleton-like fashion.
  virtual port::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) = 0;

 protected:
  // SE_DISALLOW_COPY_AND_ASSIGN declares a constructor, which suppresses the
  // presence of the default constructor. This statement re-enables it, which
  // simplifies subclassing.
  Platform() = default;

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(Platform);
};

}  // namespace stream_executor
}  // namespace tensorflow_cpy
// clang-format on

#endif  // TENSORFLOW_CPY_STREAM_EXECUTOR_PLATFORM_H_
