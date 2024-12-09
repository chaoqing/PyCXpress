#ifndef TENSORFLOW_CORE_FRAMEWORK_TENSOR_H_
#define TENSORFLOW_CORE_FRAMEWORK_TENSOR_H_

#include <cstdint>
#include <type_traits>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/framework/allocator.h"

namespace tensorflow {

// Forward declarations.  In particular, we forward declare protos so that their
// symbols can be removed from .so exports.
class AllocationDescription;

/// @ingroup core

/// Interface to access the raw ref-counted data buffer.
class TensorBuffer : public core::RefCounted {
 public:
  explicit TensorBuffer(void* data_ptr) : data_(data_ptr) {}
  ~TensorBuffer() {}

  /// \brief data() points to a memory region of size() bytes.
  ///
  /// NOTE(mrry): The `data()` method is not virtual for performance reasons.
  /// It can be called multiple times when the contents of a `Tensor` are
  /// accessed, and so making it non-virtual allows the body to be inlined.
  void* data() const { return data_; }

  /// \brief Size (in bytes) of the buffer.
  virtual size_t size() const = 0;

  /// \brief If this TensorBuffer is sub-buffer of another TensorBuffer,
  /// returns that TensorBuffer. Otherwise, returns this.
  virtual TensorBuffer* root_buffer() = 0;

  /// \brief Fills metadata about the allocation into the proto.
  virtual void FillAllocationDescription(
      AllocationDescription* proto) const = 0;

  virtual bool GetAllocatedBytes(size_t* out_bytes) const;

  /// \brief Helper method to reinterpret the buffer as an array of `T`.
  template <typename T>
  T* base() const {
    return reinterpret_cast<T*>(data());
  }

  /// \brief Whether this TensorBuffer owns the underlying memory.
  virtual bool OwnsMemory() const { return true; }

  /// \brief The type of the underlying memory.
  virtual AllocatorMemoryType GetMemoryType() const {
    return AllocatorMemoryType::kUnknown;
  }

 private:
  void* const data_;
};

/// Represents an n-dimensional array of values.
class Tensor {
 public:
  /// \brief Creates a 1-dimensional, 0-element float tensor.
  ///
  /// The returned Tensor is not a scalar (shape {}), but is instead
  /// an empty one-dimensional Tensor (shape {0}, NumElements() ==
  /// 0). Since it has no elements, it does not need to be assigned a
  /// value and is initialized by default (IsInitialized() is
  /// true). If this is undesirable, consider creating a one-element
  /// scalar which does require initialization:
  ///
  /// ```c++
  ///
  ///     Tensor(DT_FLOAT, TensorShape({}))
  ///
  /// ```
  Tensor();

  /// \brief Creates a Tensor of the given `type` and `shape`.  If
  /// LogMemory::IsEnabled() the allocation is logged as coming from
  /// an unknown kernel and step. Calling the Tensor constructor
  /// directly from within an Op is deprecated: use the
  /// OpKernelConstruction/OpKernelContext allocate_* methods to
  /// allocate a new tensor, which record the kernel and step.
  ///
  /// The underlying buffer is allocated using a `CPUAllocator`.
  Tensor(DataType type, const TensorShape& shape);

  /// \brief Creates a tensor with the input `type` and `shape`, using
  /// the allocator `a` to allocate the underlying buffer. If
  /// LogMemory::IsEnabled() the allocation is logged as coming from
  /// an unknown kernel and step. Calling the Tensor constructor
  /// directly from within an Op is deprecated: use the
  /// OpKernelConstruction/OpKernelContext allocate_* methods to
  /// allocate a new tensor, which record the kernel and step.
  ///
  /// `a` must outlive the lifetime of this Tensor.
  Tensor(Allocator* a, DataType type, const TensorShape& shape);

  /*
  /// \brief Creates a tensor with the input `type` and `shape`, using
  /// the allocator `a` and the specified "allocation_attr" to
  /// allocate the underlying buffer. If the kernel and step are known
  /// allocation_attr.allocation_will_be_logged should be set to true
  /// and LogMemory::RecordTensorAllocation should be called after the
  /// tensor is constructed. Calling the Tensor constructor directly
  /// from within an Op is deprecated: use the
  /// OpKernelConstruction/OpKernelContext allocate_* methods to
  /// allocate a new tensor, which record the kernel and step.
  ///
  /// `a` must outlive the lifetime of this Tensor.
  Tensor(Allocator* a, DataType type, const TensorShape& shape,
         const AllocationAttributes& allocation_attr);
  */

  /// \brief Creates a tensor with the input datatype, shape and buf.
  ///
  /// Acquires a ref on buf that belongs to this Tensor.
  Tensor(DataType type, const TensorShape& shape, TensorBuffer* buf);

  /// \brief Creates a tensor with the input datatype, shape and buf.
  ///
  /// Takes an ownership of the bufffer from the reference counted pointer.
  Tensor(DataType type, TensorShape shape, core::RefCountPtr<TensorBuffer> buf);

  /// \brief Creates an empty Tensor of the given data type.
  ///
  /// Like Tensor(), returns a 1-dimensional, 0-element Tensor with
  /// IsInitialized() returning True. See the Tensor() documentation
  /// for details.
  explicit Tensor(DataType type);

  /// \brief Initializes a tensor with the input `type` and `shape`, or returns
  /// an error and leaves `out_tensor` unmodified. This factory method should be
  /// used instead of the corresponding constructor if calling code cannot
  /// validate that the `DataType` is valid and supported.
  ///
  /// The underlying buffer is allocated using a `CPUAllocator`.
  static Status BuildTensor(DataType type, const TensorShape& shape,
                            Tensor* out_tensor);

 public:
  /*
  // A series of specialized constructors for scalar tensors in host memory.
  //
  // NOTE: The `Variant` host-scalar constructor is not defined, because Variant
  // is implicitly constructible from many different types, and this causes
  // ambiguities with some compilers.
  explicit Tensor(float scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(double scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(int32_t scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(uint32 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(uint16 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(uint8 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(int16_t scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(int8_t scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(tstring scalar_value)
      : Tensor(std::move(scalar_value), host_scalar_tag{}) {}
  explicit Tensor(complex64 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(complex128 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(int64_t scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(uint64 scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}
  explicit Tensor(bool scalar_value)
      : Tensor(scalar_value, host_scalar_tag{}) {}

  // NOTE: The `const char*` host-scalar constructor is provided as a
  // convenience because otherwise passing a string literal would surprisingly
  // construct a DT_BOOL tensor.
  explicit Tensor(const char* scalar_value)
      : Tensor(tstring(scalar_value), host_scalar_tag{}) {}
  */

  /// Copy constructor.
  Tensor(const Tensor& other);

  /// \brief Move constructor. After this call, <other> is safely destructible
  /// can be assigned to, and IsInitialized() can be called and will return
  /// false. Other calls on <other> (e.g. shape manipulation) are not valid.
  Tensor(Tensor&& other);

  // Explicitly delete constructor that take a pointer (except char*)
  // so that the pointer doesn't get implicitly cast to bool.
  template <typename T, typename std::enable_if<!std::is_same<T, char>::value,
                                                T>::type* = nullptr>
  explicit Tensor(T* t) = delete;

  ~Tensor();

  /// Returns the data type.
  DataType dtype() const ;

  /// Returns the shape of the tensor.
  const TensorShape& shape() const { return shape_; }

  /// \brief Convenience accessor for the tensor shape.
  ///
  /// For all shape accessors, see comments for relevant methods of
  /// `TensorShape` in `tensor_shape.h`.
  int dims() const { return shape().dims(); }

  /// Convenience accessor for the tensor shape.
  int64_t dim_size(int d) const { return shape().dim_size(d); }

  /// Convenience accessor for the tensor shape.
  int64_t NumElements() const { return shape().num_elements(); }

  bool IsSameSize(const Tensor& b) const {
    return shape().IsSameSize(b.shape());
  }

  // True iff the two tensors use the same underlying refcounted storage
  bool SharesBufferWith(const Tensor& b) const;

  /// \brief If necessary, has this Tensor been initialized?
  ///
  /// Zero-element Tensors are always considered initialized, even if they
  /// have never been assigned to and do not have any memory allocated.
  bool IsInitialized() const;

  /// Returns the estimated memory usage of this tensor.
  size_t TotalBytes() const;

  // Returns the size of allocated memory for this tensor.
  size_t AllocatedBytes() const;

  /// Returns true iff this tensor is aligned.
  bool IsAligned() const {
#if EIGEN_MAX_ALIGN_BYTES == 0
    return true;
#else
    void* ptr = base<void>();
    return dtype() == DT_STRING || NumElements() == 0 ||
           (reinterpret_cast<intptr_t>(ptr) % EIGEN_MAX_ALIGN_BYTES == 0);
#endif
  }

  /// Assign operator. This tensor shares other's underlying storage.
  Tensor& operator=(const Tensor& other) {
    CopyFromInternal(other, other.shape());
    return *this;
  }

  /// Move operator.  See move constructor for details.
  Tensor& operator=(Tensor&& other);

  /// \brief Copy the other tensor into this tensor and reshape it.
  ///
  /// This tensor shares other's underlying storage. Returns `true`
  /// iff `other.shape()` has the same number of elements of the given
  /// `shape`.
  bool CopyFrom(const Tensor& other,
                const TensorShape& shape) TF_MUST_USE_RESULT {
    if (other.NumElements() != shape.num_elements()) return false;
    CopyFromInternal(other, shape);
    return true;
  }

  /// \brief Slice this tensor along the 1st dimension.

  /// I.e., the returned tensor satisfies
  ///     returned[i, ...] == this[dim0_start + i, ...].
  /// The returned tensor shares the underlying tensor buffer with this
  /// tensor.
  ///
  /// NOTE: The returned tensor may not satisfy the same alignment
  /// requirement as this tensor depending on the shape. The caller
  /// must check the returned tensor's alignment before calling certain
  /// methods that have alignment requirement (e.g., `flat()`, `tensor()`).
  ///
  /// NOTE: When fed with an N-dimensional tensor, this method returns a tensor
  /// also with N dimensions. If you want to select a sub tensor, see SubSlice.
  ///
  /// REQUIRES: `dims()` >= 1
  /// REQUIRES: `0 <= dim0_start <= dim0_limit <= dim_size(0)`
  Tensor Slice(int64_t dim0_start, int64_t dim0_limit) const;

  /// \brief Select a subslice from this tensor along the 1st dimension.
  ///
  /// When fed with an N-dimensional tensor, this method returns a tensor with
  /// N-1 dimensions, where the returned tensor is a subslice of the input
  /// tensor along the first dimension. The N-1 dimensions of the returned
  /// tensor are the last N-1 dimensions of the input tensor.
  ///
  /// NOTE: The returned tensor may not satisfy the same alignment
  /// requirement as this tensor depending on the shape. The caller
  /// must check the returned tensor's alignment before calling certain
  /// methods that have alignment requirement (e.g., `flat()`, `tensor()`).
  ///
  /// REQUIRES: `dims()` >= 1
  /// REQUIRES: `0 <= index < dim_size(0)`
  Tensor SubSlice(int64_t index) const;

  /// Render the first `max_entries` values in `*this` into a string.
  std::string SummarizeValue(int64_t max_entries, bool print_v2 = false) const;

  /// A human-readable summary of the tensor suitable for debugging.
  // `num_values` is the number of actual data values in the tensor
  // included in the message. If the tensor might be resident in
  // GPU/TPU memory use DeviceSafeDebugString instead.
  std::string DebugString(int num_values) const;
  std::string DebugString() const { return DebugString(3); }

  // Variant of DebugString() that should be used for possibly non-CPU tensors.
  // If the tensor is not resident on CPU, we can't read its values as
  // DebugString() does.
  std::string DeviceSafeDebugString() const;

  /// \brief Returns a `StringPiece` mapping the current tensor's buffer.
  ///
  /// The returned `StringPiece` may point to memory location on devices
  /// that the CPU cannot address directly.
  ///
  /// NOTE: The underlying tensor buffer is refcounted, so the lifetime
  /// of the contents mapped by the `StringPiece` matches the lifetime of
  /// the buffer; callers should arrange to make sure the buffer does
  /// not get destroyed while the `StringPiece` is still used.
  ///
  /// REQUIRES: `DataTypeCanUseMemcpy(dtype())`.
  void* data() const;


  // Returns true if the refcount on buf_ and any possible underlying root
  // buffer is one.
  bool RefCountIsOne() const;

  // Returns the type of the underlying memory.
  AllocatorMemoryType GetMemoryType() const { return buf_->GetMemoryType(); }

 private:

  TensorShape shape_;
  TensorBuffer* buf_;

  void CopyFromInternal(const Tensor& other, const TensorShape& shape) ;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_TENSOR_H_
