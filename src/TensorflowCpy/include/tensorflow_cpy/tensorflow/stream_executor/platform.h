// Defines types and declares functions for identifying and extracting
// information about the types of platforms and supporting libraries for which
// StreamExecutor implementations exist.
#ifndef TENSORFLOW_STREAM_EXECUTOR_PLATFORM_H_
#define TENSORFLOW_STREAM_EXECUTOR_PLATFORM_H_

#include <map>

//#include "tensorflow/stream_executor/device_options.h"
//#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/core/platform/statusor.h"
//#include "tensorflow/stream_executor/platform/port.h"
//#include "tensorflow/stream_executor/plugin.h"
//#include "tensorflow/stream_executor/trace_listener.h"


namespace stream_executor {
//#include "tensorflow/stream_executor/device_description.h"
// Data that describes the execution target of the StreamExecutor, in terms of
// important logical parameters. These include dimensionality limits and
// physical parameters of interest, such as number of cores present on the
// device.
//
// Thread-safe: immutable post-initialization.
class DeviceDescription {
 public:
  // Returns the platform being run on; this value is primarily intended for
  // printing, and comes out something like "OpenCL 1.2" or "Compute Capability
  // 3.5".
  const std::string &platform_version() const { return platform_version_; }

  // Returns the driver version interfacing with the underlying platform. Vendor
  // dependent format.
  const std::string &driver_version() const { return driver_version_; }

  // Return the runtime version, if one is provided by the underlying platform.
  // Vendor dependent format / usefulness.
  const std::string &runtime_version() const { return runtime_version_; }

  // Returns the name that the device reports. Vendor dependent.
  const std::string &name() const { return name_; }

  // Returns the PCI bus identifier for this device, of the form
  // [domain]:[bus]:[device].[function]
  const std::string &pci_bus_id() const { return pci_bus_id_; }

  // Returns the NUMA node associated with this device, for use in
  // determining socket locality. If the NUMA node could not be determined, -1
  // is returned.
  int numa_node() const { return numa_node_; }

  // Number of cores (traditional notion of core; i.e. an SM on an NVIDIA device
  // or an AMD Compute Unit.
  int core_count() const { return core_count_; }

  // Returns the limit on the total number of threads that can be launched in a
  // single block; i.e. the limit on x * y * z dimensions of a ThreadDim.
  // This limit affects what constitutes a legitimate kernel launch request.
  const int64_t &threads_per_block_limit() const {
    return threads_per_block_limit_;
  }

  // Returns the limit on the total number of threads that can be simultaneously
  // launched on a given multiprocessor.
  const int64_t &threads_per_core_limit() const {
    return threads_per_core_limit_;
  }

  // Returns the number of threads per warp/wavefront.
  const int64_t &threads_per_warp() const { return threads_per_warp_; }

  // Returns the limit on the total number of registers per core.
  const int64_t &registers_per_core_limit() const {
    return registers_per_core_limit_;
  }

  // Returns the limit on the total number of registers that can be
  // simultaneously used by a block.
  const int64_t &registers_per_block_limit() const {
    return registers_per_block_limit_;
  }

  // Returns the number of address bits available to kernel code running on the
  // platform. This affects things like the maximum allocation size and perhaps
  // types used in kernel code such as size_t.
  const int64_t &device_address_bits() const { return device_address_bits_; }

  // Returns the device memory size in bytes.
  int64_t device_memory_size() const { return device_memory_size_; }

  // Returns the device's memory bandwidth in bytes/sec.  (This is for
  // reads/writes to/from the device's own memory, not for transfers between the
  // host and device.)
  int64_t memory_bandwidth() const { return memory_bandwidth_; }

  // Returns the device's core clock rate in GHz.
  float clock_rate_ghz() const { return clock_rate_ghz_; }

  // Returns whether ECC is enabled.
  bool ecc_enabled() const { return ecc_enabled_; }

  // Returns the device vendor string, e.g., "NVIDIA Corporation", "Advanced
  // Micro Devices, Inc.", or "GenuineIntel".
  const std::string &device_vendor() const { return device_vendor_; }

  //// Returns the CUDA compute capability if we're running on the CUDA platform.
  //// If a CUDA compute capability is not available, the major version will be
  //// zero.
  //CudaComputeCapability cuda_compute_capability() const;


  // Returns the maximum amount of shared memory present on a single core
  // (i.e. Streaming Multiprocessor on NVIDIA GPUs; Compute Unit for OpenCL
  // devices). Note that some devices, such as NVIDIA's have a configurable
  // partitioning between shared memory and L1 cache.
  int64_t shared_memory_per_core() const { return shared_memory_per_core_; }

  // Returns the maximum amount of shared memory available for a single block.
  int64_t shared_memory_per_block() const { return shared_memory_per_block_; }

  // TODO(leary): resident blocks per core will be useful.

  // Convenience typedef for the string-based DeviceDescription mapping.
  typedef std::map<std::string, std::string> Map;

  // Returns a mapping from readable names to readable values that describe the
  // device. This is useful for things like printing.
  std::unique_ptr<Map> ToMap() const;

  // For string values that are not available via the underlying platform, this
  // value will be provided.
  static const char *kUndefinedString;

 private:
  DeviceDescription();

  // For description of the following members, see the corresponding accessor
  // above.
  //
  // N.B. If another field is added, update ToMap() above.
  std::string device_vendor_;
  std::string platform_version_;
  std::string driver_version_;
  std::string runtime_version_;
  std::string pci_bus_id_;
  std::string name_;

  int64_t threads_per_core_limit_;
  int64_t threads_per_block_limit_;
  int64_t threads_per_warp_;

  int64_t registers_per_core_limit_;
  int64_t registers_per_block_limit_;

  int64_t device_address_bits_;
  int64_t device_memory_size_;
  int64_t memory_bandwidth_;

  // Shared memory limits on a given device.
  int64_t shared_memory_per_core_;
  int64_t shared_memory_per_block_;

  float clock_rate_ghz_;

  //// CUDA "CC" major value, -1 if not available.
  //CudaComputeCapability cuda_compute_capability_{-1, -1};

  int numa_node_;
  int core_count_;
  bool ecc_enabled_;

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

// Returns true if kind represents a valid platform capable of enqueuing items
// on a stream, but not necessarily on an accelerator device.
// Returns false for kMock and any invalid PlatformKind values.
bool PlatformIsRunnable(PlatformKind kind);

// Returns true if kind represents a valid platform capable of running kernels
// on an accelerator device. Returns false for kHost*, kMock and any invalid
// PlatformKind values.
bool PlatformIsRunnableOnDevice(PlatformKind kind);

// Returns a printable description of a PlatformKind.
std::string PlatformKindString(PlatformKind kind);

// Returns the PlatformKind corresponding to the input string; returns kInvalid
// in the case of no match.
PlatformKind PlatformKindFromString(std::string platform_string);

// Checks that kind takes on a valid value.
void CheckPlatformKindIsValid(PlatformKind kind);

// StreamExecutorConfig encapsulates the set of options for constructing a
// StreamExecutor for a given platform.
struct StreamExecutorConfig {
  // Sets members to defaults: -1 for ordinal (must be changed), and default
  // PluginConfig and DeviceOptions.
  StreamExecutorConfig();

  // Simple ordinal-setting constructor.
  explicit StreamExecutorConfig(int ordinal);

  // The GPU stream for which we are searching the executor.
  // If this field is specified for the search, others will be ignored.
  void* gpu_stream = nullptr;

  // The ordinal of the device to be managed by the returned StreamExecutor.
  int ordinal;

  //// The PluginConfig for the returned StreamExecutor.
  //PluginConfig plugin_config;

  //// The DeviceOptions for the returned StreamExecutor.
  //DeviceOptions device_options;
};

// Abstract base class for a platform registered with the MultiPlatformManager.
class Platform {
 public:
  virtual ~Platform();

  // A platform ID is a unique identifier for each registered platform type -
  // each platform is required to expose an ID to ensure unique registration and
  // as a target against which plugins can register.
  //
  // The macro below is provided to help generate a [process-unique] identifier.
  using Id = void*;

// Helper macro to define a plugin ID. To be used only inside plugin
// implementation files. Works by "reserving" an address/value (guaranteed to be
// unique) inside a process space.
#define PLATFORM_DEFINE_ID(ID_VAR_NAME) \
  namespace {                           \
  int plugin_id_value;                  \
  }                                     \
  const ::stream_executor::Platform::Id ID_VAR_NAME = &plugin_id_value;

  // Returns a key uniquely identifying this platform.
  virtual Id id() const = 0;

  // Name of this platform.
  virtual const std::string& Name() const = 0;

  // Returns the number of devices accessible on this platform.
  //
  // Note that, though these devices are visible, if there is only one userspace
  // context allowed for the device at a time and another process is using this
  // device, a call to ExecutorForDevice may return an error status.
  virtual int VisibleDeviceCount() const = 0;

  // Returns true iff the platform has been initialized.
  virtual bool Initialized() const;

  // Initializes the platform with a custom set of options. The platform must be
  // initialized before obtaining StreamExecutor objects.  The interpretation of
  // the platform_options argument is implementation specific.  This method may
  // return an error if unrecognized options are provided.  If using
  // MultiPlatformManager, this method will be called automatically by
  // InitializePlatformWithId/InitializePlatformWithName.
  virtual port::Status Initialize(
      const std::map<std::string, std::string>& platform_options);

  //// Returns a populated DeviceDescription for the device at the given ordinal.
  //// This should not require device initialization. Note that not all platforms
  //// may support acquiring the DeviceDescription indirectly.
  ////
  //// Alternatively callers may call GetDeviceDescription() on the StreamExecutor
  //// which returns a cached instance specific to the initialized StreamExecutor.
  //virtual port::StatusOr<std::unique_ptr<DeviceDescription>>
  //DescriptionForDevice(int ordinal) const = 0;

  // Returns a device with the given ordinal on this platform with a default
  // plugin configuration or, if none can be found with the given ordinal or
  // there is an error in opening a context to communicate with the device, an
  // error status is returned.
  //
  // Ownership of the executor is NOT transferred to the caller --
  // the Platform owns the executors in a singleton-like fashion.
  virtual port::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) = 0;

  //// Returns a device or error, as above, with the specified plugins.
  ////
  //// Ownership of the executor is NOT transferred to the caller.
  //virtual port::StatusOr<StreamExecutor*> ExecutorForDeviceWithPluginConfig(
      //int ordinal, const PluginConfig& plugin_config) = 0;

  //// Returns a device constructed with the options specified in "config".
  //// Ownership of the executor is NOT transferred to the caller.
  //virtual port::StatusOr<StreamExecutor*> GetExecutor(
      //const StreamExecutorConfig& config) = 0;

  //// Returns a device constructed with the options specified in "config" without
  //// looking in or storing to the Platform's executor cache.
  //// Ownership IS transferred to the caller.
  //virtual port::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      //const StreamExecutorConfig& config) = 0;

  // Warning: this is a dangerous API and should be used with caution.
  //
  // Forces the platform to delete executor instances, releasing their
  // associated device contexts. There must be no held instances of the executor
  // and there must be no outstanding activity on the devices for this platform.
  //
  // This is only useful on platforms which bind a device to a single process
  // that has obtained the device context. May return UNIMPLEMENTED on platforms
  // that have no reason to destroy device contexts.
  //
  // The platform must be reinitialized after this is called.
  virtual port::Status ForceExecutorShutdown();


  // Map of executor-to-executor coordinate and boolean, indicating if the first
  // executor can access the second's memory.
  using PeerAccessMap = std::map<std::pair<int, int>, bool>;

  // Returns a matrix indicating which executors can access which other
  // executors' memory.
  virtual std::unique_ptr<PeerAccessMap> GetPeerAccessMap();

  // Attempts to enable all peer-to-peer access links described by the result of
  // GetPeerAccessMap(). Note that calling this routine will force the creation
  // of a default-argument (see StreamExecutorConfig) StreamExecutor object for
  // each device ordinal in the system, should any not yet exist.
  virtual port::Status EnablePeerAccess();

 protected:
  // SE_DISALLOW_COPY_AND_ASSIGN declares a constructor, which suppresses the
  // presence of the default constructor. This statement re-enables it, which
  // simplifies subclassing.
  Platform() = default;

 private:
  SE_DISALLOW_COPY_AND_ASSIGN(Platform);
};

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_PLATFORM_H_
