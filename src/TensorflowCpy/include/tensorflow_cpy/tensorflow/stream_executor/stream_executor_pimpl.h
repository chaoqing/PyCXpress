
#ifndef TENSORFLOW_CPY_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_
#define TENSORFLOW_CPY_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_


#include "../stream_executor/device_memory.h"
#include "../stream_executor/platform.h"

// clang-format off
namespace tensorflow_cpy {
namespace stream_executor {
  using namespace ::tensorflow;

  namespace internal{
    class StreamExecutorInterface{};
  };


// A StreamExecutor manages a single device, in terms of executing work (kernel
// launches) and memory management (allocation/deallocation, memory copies to
// and from the device). It is conceptually the "handle" for a device -- Stream
// objects, which are used to enqueue work to run on the
// coprocessor have a StreamExecutor instance as their "parent" object.
//
// StreamExecutor objects have an underlying platform that is specified up
// front;
// e.g. either it is a CUDA or OpenCL executor.
//
// Thread-safe after initialization.
// StreamExecutor interface should not be invoked from a signal handler.
class StreamExecutor {
 public:
  StreamExecutor(
      const Platform* platform,
      std::unique_ptr<internal::StreamExecutorInterface> implementation,
      int device_ordinal);

  ~StreamExecutor();

  port::Status Init();

  // Returns the platform that this StreamExecutor is acting upon.
  PlatformKind platform_kind() const ;

  // Returns a reference to the platform that created this executor.
  const Platform* platform() const ;

  // Synchronizes all activity occurring in the StreamExecutor's context (most
  // likely a whole device).
  bool SynchronizeAllActivity() SE_MUST_USE_RESULT;

  // Blocks the caller while "size" bytes are zeroed out (in POD fashion) at the
  // given location in device memory.
  port::Status SynchronousMemZero(DeviceMemoryBase* location,
                                  uint64_t size) SE_MUST_USE_RESULT;

  // Blocks the caller while "size" bytes are initialized to "value" (in POD
  // fashion) at the given location in device memory.
  port::Status SynchronousMemSet(DeviceMemoryBase* location, int value,
                                 uint64_t size) SE_MUST_USE_RESULT;

  // [deprecated] Blocks the caller while a data segment of the given size is
  // copied from the host source to the device destination.
  bool SynchronousMemcpy(DeviceMemoryBase* device_dst, const void* host_src,
                         uint64_t size) SE_MUST_USE_RESULT;

  // [deprecated] Blocks the caller while a data segment of the given size is
  // copied from the device source to the host destination.
  bool SynchronousMemcpy(void* host_dst, const DeviceMemoryBase& device_src,
                         uint64_t size) SE_MUST_USE_RESULT;

  // Same as SynchronousMemcpy(DeviceMemoryBase*, ...) above.
  port::Status SynchronousMemcpyH2D(const void* host_src, int64_t size,
                                    DeviceMemoryBase* device_dst);

  // Same as SynchronousMemcpy(void*, ...) above.
  port::Status SynchronousMemcpyD2H(const DeviceMemoryBase& device_src,
                                    int64_t size, void* host_dst);

  // Blocks the caller while a data segment of the given size is copied from the
  // device source to the device destination.
  bool SynchronousMemcpy(DeviceMemoryBase* device_dst,
                         const DeviceMemoryBase& device_src,
                         uint64_t size) SE_MUST_USE_RESULT;

  // Obtains metadata about the underlying device.
  // The value is cached on first use.
  const DeviceDescription& GetDeviceDescription() const;

  // If implemented, returns device specific measurement of load
  // (e.g. pending requests).
  int64_t GetDeviceLoad() const;

  // Returns the underlying device memory usage information, if it is available.
  // If it is not available (false is returned), free/total may not be
  // initialized.
  //
  // Note: "Free" reflects the amount of free memory on the underlying device,
  // so allocations via other StreamExecutors that have the same underlying
  // device
  // will be reflected in "free".
  bool DeviceMemoryUsage(int64_t* free, int64_t* total) const;

  // The device count reported by this StreamExecutor's platform.
  // Note: on OpenCL we implicitly select platform zero at the moment.
  int PlatformDeviceCount() const;

 private:

  // Reference to the platform that created this executor.
  const Platform* platform_;

  SE_DISALLOW_COPY_AND_ASSIGN(StreamExecutor);
};

}  // namespace stream_executor
}  // namespace tensorflow_cpy
// clang-format on

#endif  // TENSORFLOW_CPY_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_
