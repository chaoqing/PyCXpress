// Suite of types that represent device memory allocations. These are
// allocated by the StreamExecutor interface, which produces values appropriate
// for the underlying platform (whether it be CUDA or OpenCL).
//
// The untyped base class (like a device void*) is DeviceMemoryBase, which can
// be specialized for a given allocation type (like a device T*) using
// DeviceMemory<T>.

#ifndef TENSORFLOW_CPY_STREAM_EXECUTOR_DEVICE_MEMORY_H_
#define TENSORFLOW_CPY_STREAM_EXECUTOR_DEVICE_MEMORY_H_

#include <stddef.h>

#include <cstdint>


// clang-format off
namespace tensorflow_cpy {
namespace stream_executor {

class DeviceMemoryAllocator;
class StreamExecutor;

// void*-analogous device memory allocation. For the typed variation, see
// DeviceMemory<T>.
//
// This is effectively a two-tuple of a pointer and size; however, note that the
// pointer may not be to the virtual address itself -- in OpenCL the pointer is
// to a cl_mem handle that describes the device allocation. Therefore,
// DeviceMemoryBase::opaque does not necessarily produce a pointer that can be
// referenced directly, so use it with caution.
//
// Thread-compatible.
class DeviceMemoryBase {
 public:
  // Default constructor instantiates a null-pointed, zero-sized device memory
  // region. An opaque pointer may be provided -- see header for details on the
  // opacity of that pointer.
  explicit DeviceMemoryBase(void *opaque = nullptr, uint64_t size = 0)
      : opaque_(opaque), size_(size) {}

  // Returns whether the backing memory is the null pointer.
  // A `== nullptr` convenience method is also provided.
  bool is_null() const { return opaque_ == nullptr; }
  bool operator==(std::nullptr_t ) const { return is_null(); }
  bool operator!=(std::nullptr_t ) const { return !is_null(); }

  // Provides a partial order between device memory values.
  //
  // This operator is provided so that this object can be used as a key in an
  // ordered map.
  bool operator<(const DeviceMemoryBase &other) const {
    return opaque() < other.opaque();
  }

  // Returns the size, in bytes, for the backing memory.
  uint64_t size() const { return size_; }

  // Warning: note that the pointer returned is not necessarily directly to
  // device virtual address space, but is platform-dependent.
  void *opaque() { return opaque_; }
  const void *opaque() const { return opaque_; }

  // Returns the payload of this memory region.
  uint64_t payload() const { return payload_; }

  // Sets payload to given value.
  void SetPayload(uint64_t payload) { payload_ = payload; }

  // Returns whether the two DeviceMemoryBase segments are identical (both in
  // their opaque pointer and size).
  bool IsSameAs(const DeviceMemoryBase &other) const {
    return opaque() == other.opaque() && size() == other.size();
  }

 protected:
  friend class StreamExecutor;

  // Resets the internal values of the opaque pointer and number of bytes in the
  // memory region, just as in the constructor.
  void Reset(void *opaque, uint64_t bytes) {
    opaque_ = opaque;
    size_ = bytes;
  }

 private:
  void *opaque_;  // Platform-dependent value representing allocated memory.
  uint64_t size_;         // Size in bytes of this allocation.
  uint64_t payload_ = 0;  // Payload data associated with this allocation.
};

// Typed wrapper around "void *"-like DeviceMemoryBase.
//
// For example, DeviceMemory<int> is a simple wrapper around DeviceMemoryBase
// that represents one or more integers in Device memory.
//
// Thread-compatible.
template <typename ElemT>
class DeviceMemory final : public DeviceMemoryBase {
 public:
  // Default constructor instantiates a null-pointed, zero-sized memory region.
  DeviceMemory() : DeviceMemoryBase(nullptr, 0) {}
  explicit DeviceMemory(std::nullptr_t) : DeviceMemory() {}

  // Typed device memory regions may be constructed from untyped device memory
  // regions, this effectively amounts to a cast from a void*.
  explicit DeviceMemory(const DeviceMemoryBase &other)
      : DeviceMemoryBase(const_cast<DeviceMemoryBase &>(other).opaque(),
                         other.size()) {
    SetPayload(other.payload());
  }

  // Returns the number of elements of type ElemT that constitute this
  // allocation.
  uint64_t ElementCount() const { return size() / sizeof(ElemT); }

  // Returns whether this is a single-element allocation.
  bool IsScalar() const { return ElementCount() == 1; }

  // Create a typed area of DeviceMemory with a given opaque pointer and the
  // quantity of bytes in the allocation. This function is broken out to
  // distinguish bytes from an element count.
  static DeviceMemory<ElemT> MakeFromByteSize(void *opaque, uint64_t bytes) {
    return DeviceMemory<ElemT>(opaque, bytes);
  }

  // Resets the DeviceMemory data, in MakeFromByteSize fashion.
  // This simply clobbers the prior values.
  void ResetFromByteSize(void *opaque, uint64_t bytes) {
    // TODO(leary) when NVCC is eliminated we can add this check (and the
    // logging include it requires).
    // CHECK_EQ(0, bytes % sizeof(ElemT));
    DeviceMemoryBase::Reset(opaque, bytes);
  }

  // ------------------------------------------------------------

 protected:
  // This constructor is solely used from derived classes; it is made protected
  // because it accepts a byte-size instead of an element count, which could
  // potentially be misused given the ElementCount() nature of this interface.
  //
  // In order to specify the desire to use byte size instead of element count
  // explicitly, use MakeFromByteSize.
  DeviceMemory(void *opaque, uint64_t size) : DeviceMemoryBase(opaque, size) {}
};

}  // namespace stream_executor
}  // namespace tensorflow_cpy
// clang-format on

#endif  // TENSORFLOW_CPY_STREAM_EXECUTOR_DEVICE_MEMORY_H_
