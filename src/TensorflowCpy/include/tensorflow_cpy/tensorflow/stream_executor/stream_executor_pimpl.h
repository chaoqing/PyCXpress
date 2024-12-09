
#ifndef TENSORFLOW_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_
#define TENSORFLOW_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_


#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/stream_executor/platform.h"

namespace stream_executor {


class Stream;

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
  //StreamExecutor(
      //const Platform* platform,
      //std::unique_ptr<internal::StreamExecutorInterface> implementation,
      //int device_ordinal);

  ~StreamExecutor();

  port::Status Init();
  //port::Status Init(DeviceOptions device_options);

  // Returns the platform that this StreamExecutor is acting upon.
  PlatformKind platform_kind() const ;

  // Returns a reference to the platform that created this executor.
  const Platform* platform() const ;


  port::StatusOr<std::shared_ptr<DeviceMemoryBase>> CreateOrShareConstant(
      Stream* stream, const std::vector<uint8_t>& content);

  // Synchronously allocates an array on the device of type T with element_count
  // elements.
  template <typename T>
  DeviceMemory<T> AllocateArray(uint64_t element_count,
                                int64_t memory_space = 0);

  // Convenience wrapper that allocates space for a single element of type T in
  // device memory.
  template <typename T>
  DeviceMemory<T> AllocateScalar() {
    return AllocateArray<T>(1);
  }

  // Synchronously allocates a scalar of type T on the device that is (POD)
  // zero-byte initialized.
  template <typename T>
  DeviceMemory<T> AllocateZeroed();


  // Allocate a memory region inside another allocated memory region.
  // Offset and size are specified in terms of T elements.
  // Warning: Do not free a parent buffer before its sub-buffers; this may cause
  // use-after-free issues (the specific behavior is not consistent across
  // platforms).
  //  - Note: OpenCL uses refcounting to manage buffer lifetimes, so use of a
  //    sub-buffer after parent deallocation is expected to be safe. This will
  //    render your code non-platform-portable, however.
  template <typename T>
  DeviceMemory<T> GetSubBuffer(DeviceMemory<T>* parent, uint64_t element_offset,
                               uint64_t element_count);

  // Deallocate the DeviceMemory previously allocated via this interface.
  // Deallocation of a nullptr-representative value is permitted.
  //
  // Resets the internal contents of mem to be null-representative, but this
  // null-out effect should not be relied upon in client code.
  void Deallocate(DeviceMemoryBase* mem);

  // Allocates unified memory space of the given size, if supported.
  // See
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-unified-memory-programming-hd
  // for more details on unified memory.
  void* UnifiedMemoryAllocate(uint64_t bytes);

  // Deallocates unified memory space previously allocated with
  // UnifiedMemoryAllocate.
  void UnifiedMemoryDeallocate(void* location);

  // Allocates a region of host memory and registers it with the platform API.
  // Memory allocated in this manner (or allocated and registered with
  // HostMemoryRegister() is required for use in asynchronous memcpy operations,
  // such as Stream::ThenMemcpy.
  void* HostMemoryAllocate(uint64_t size);

  // Deallocates a region of host memory allocated by HostMemoryAllocate().
  void HostMemoryDeallocate(void* location);

  // Registers a region of host memory with the platform API. Registered memory
  // (or memory allocated with HostMemoryAllocate) is required for use with
  // asynchronous memcpy operations, such as Stream::ThenMemcpy. This method
  // is used to register memory allocated outside the StreamExecutor;
  // HostMemoryAllocate implicitly registers its allocations and
  // HostMemoryDeallocate implicitly deregisters on deallocation.
  bool HostMemoryRegister(void* location, uint64_t size) SE_MUST_USE_RESULT;

  // Unregisters a region of host memory registered with HostMemoryRegister.
  // This should be done before deallocating the region with delete[]/free/etc.
  bool HostMemoryUnregister(void* location) SE_MUST_USE_RESULT;

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

  // Enqueues an operation onto stream to zero out size bytes at the given
  // device memory location. Neither stream nor location may be null. Returns
  // whether the operation was successfully enqueued onto the stream.
  port::Status MemZero(Stream* stream, DeviceMemoryBase* location,
                       uint64_t size) SE_MUST_USE_RESULT;

  // Enqueues an operation onto stream to set 32-bit patterns starting at
  // location, for byte count given by size. size must be 32-bit quantified
  // (i.e. evently divisible by 4). Returns whether the operation was
  // successfully enqueued onto the stream.
  port::Status Memset32(Stream* stream, DeviceMemoryBase* location,
                        uint32_t pattern, uint64_t size);

  // Enables peer access from this StreamExecutor to memory
  // allocated by other, such that launched device code, memcpies, etc may
  // access it directly.
  //
  // Both this StreamExecutor and other must be backed by the same platform (as
  // in
  // CUDA vs OpenCL) implementation.
  port::Status EnablePeerAccessTo(StreamExecutor* other);

  // Returns whether it's possible to enable peer access from this
  // StreamExecutor
  // to memory allocated by another.
  //
  // Even when this returns true, EnablePeerAccessTo may fail for other reasons;
  // this is more an up-front test as to whether it's expressly forbidden.
  bool CanEnablePeerAccessTo(StreamExecutor* other);

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

  // Returns whether the StreamExecutor supports BLAS routines for the platform
  // that underlies this interface.
  bool SupportsBlas() const;

  // Returns whether the StreamExecutor supports FFT routines for the platform
  // that underlies this interface.
  bool SupportsFft() const;

  // Returns whether the StreamExecutor supports RNG routines for the platform
  // that underlies this interface.
  bool SupportsRng() const;

  // Returns whether the StreamExecutor support neural net routines for the
  // platform that underlies this interface.
  bool SupportsDnn() const;


  // Returns a stream allocated by this executor, or nullptr if not found.
  // Performs linear search over alive GPU streams.
  Stream* FindAllocatedStream(void* gpu_stream) ;

 private:

  // Synchronously allocates size bytes on the underlying platform and returns
  // a DeviceMemoryBase representing that allocation. In the case of failure,
  // nullptr is returned.
  DeviceMemoryBase Allocate(uint64_t size, int64_t memory_space);
  // Entrains a memcpy operation onto stream, with a host destination location
  // host_dst and a device memory source, with target size size.
  bool Memcpy(Stream* stream, void* host_dst,
              const DeviceMemoryBase& device_src, uint64_t size);

  // Entrains a memcpy operation onto stream, with a device destination location
  // and a host memory source, with target size size.
  bool Memcpy(Stream* stream, DeviceMemoryBase* device_dst,
              const void* host_src, uint64_t size);

  // Entrains a memcpy operation onto stream, with a device destination location
  // and a device source location, with target size size. Peer access should
  // have been enabled between the StreamExecutors owning the device memory
  // regions.
  bool MemcpyDeviceToDevice(Stream* stream, DeviceMemoryBase* device_dst,
                            const DeviceMemoryBase& device_src, uint64_t size);

  // Entrains on a stream a user-specified function to be run on the host.
  // See Stream::ThenDoHostCallback for full details.
  bool HostCallback(Stream* stream, std::function<void()> callback);

  // Entrains on a stream a user-specified function to be run on the host.
  // See Stream::ThenDoHostCallback for full details.
  // This is the preferred form for a callback that may return an error.
  bool HostCallback(Stream* stream, std::function<port::Status()> callback);

  // Allocates stream resources on the underlying platform and initializes its
  // internals.
  bool AllocateStream(Stream* stream);

  // Deallocates stream resources on the underlying platform.
  void DeallocateStream(Stream* stream);

  // Causes dependent to not begin execution until other has finished its
  // last-enqueued work.
  bool CreateStreamDependency(Stream* dependent, Stream* other);

  // Reference to the platform that created this executor.
  const Platform* platform_;

  SE_DISALLOW_COPY_AND_ASSIGN(StreamExecutor);
};

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_STREAM_EXECUTOR_PIMPL_H_
