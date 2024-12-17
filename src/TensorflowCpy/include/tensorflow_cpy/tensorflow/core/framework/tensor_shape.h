#ifndef TENSORFLOW_CPY_CORE_FRAMEWORK_TENSOR_SHAPE_H_
#define TENSORFLOW_CPY_CORE_FRAMEWORK_TENSOR_SHAPE_H_

#include <string>

#include "../../core/platform/logging.h"
#include "../../core/platform/status.h"

// clang-format off
namespace tensorflow_cpy {
namespace tensorflow {
  using namespace ::tensorflow;

// START_SKIP_DOXYGEN
template <class Shape>
class TensorShapeIter;
class TensorShape;
// END_SKIP_DOXYGEN


/// Represents the shape of a Tensor.
///
/// A tensor's shape is denoted by its number of dimensions and a size for each
/// dimension.  For example, a Tensor represented by a 3 x 4 matrix would have
/// a shape of 2-D, [3,4].
///
/// If you know the exact shape of your Tensor when you create the TensorShape
/// object, you can specify it then, or you can create a TensorShape with
/// zero dimensions and one element, and call AddDim() to add dimensions later.
class TensorShape {
 public:
  ///// \brief Construct a `TensorShapeBase` from the provided sizes.
  ///// REQUIRES: `dim_sizes[i] >= 0` (or >= -1 for PartialTensorShape)
  TensorShape(std::initializer_list<int64_t> dim_sizes);

  /// Construct an empty TensorShape, or an unknown rank PartialTensorShape
  TensorShape();

  /// \brief Returns the number of elements in the tensor.
  ///
  /// We use `int64` and not `size_t` to be compatible with `Eigen::Tensor`
  /// which uses `ptrdiff_t`.  For PartialTensorShape, -1 means not fully
  /// defined.
  int64_t num_elements() const ;

  /// Returns `true` iff this is a valid tensor shape.
  bool IsValid();

  /// \brief Add a dimension to the end ("inner-most").
  /// REQUIRES: `size >= 0`
  void AddDim(int64_t size);

  /// Same as `AddDim` but returns a `Status`.
  /// Use if unsure is `size >= 0`, to prevent `CHECK`-crashes.
  Status AddDimWithStatus(int64_t size);

  /// Appends all the dimensions from `shape`.
  void AppendShape(const TensorShape& shape);

  /// Same as `RemoveDim` but returns a `Status`.
  /// Use if you cannot validate all invariants, to prevent `CHECK`-fail.
  Status AppendShapeWithStatus(const TensorShape& shape);

  /// \brief Insert a dimension somewhere in the `TensorShape`.
  /// REQUIRES: `0 <= d <= dims()`
  /// REQUIRES: `size >= 0`
  void InsertDim(int d, int64_t size);

  /// Same as `InsertDim` but returns a `Status`.
  /// Use if unsure if requirements in `InsertDim` are satistified, to prevent
  /// `CHECK`-fail crashes.
  Status InsertDimWithStatus(int d, int64_t size);

  /// \brief Modifies the size of the dimension `d` to be `size`
  /// REQUIRES: `0 <= d < dims()`
  /// REQUIRES: `size >= 0`
  void set_dim(int d, int64_t size);

  /// Same as `set_dim` but returns a `Status`.
  /// Use if unsure if requirements in `set_dim` are satistified, to prevent
  /// `CHECK`-fail crashes.
  Status SetDimWithStatus(int d, int64_t size);

  /// \brief Removes dimension `d` from the `TensorShape`.
  /// REQUIRES: `0 <= d < dims()`
  void RemoveDim(int d) {
    CHECK_GE(d, 0);
    RemoveDimRange(d, d + 1);
  }

  /// Same as `RemoveDim` but returns a `Status`.
  /// Use if unsure is `0 <= d < dims()`, to prevent `CHECK`-crashes.
  Status RemoveDimWithStatus(int64_t d) ;

  /// \brief Removes last `n` dimensions from the `TensorShape`.
  /// REQUIRES: `0 <= n <= dims()`
  void RemoveLastDims(int n) {
    CHECK_LE(n, dims());
    RemoveDimRange(dims() - n, dims());
  }

  /// Same as `RemoveLastDims` but returns a `Status`.
  /// Use if unsure is `0 <= n <= dims()`, to prevent `CHECK`-crashes.
  Status RemoveLastDimsWithStatus(int64_t n) ;

  /// \brief Removes the dimensions in range `[begin:end)` from `TensorShape`.
  /// Negative values of `end` are interpreted as `dims() + end + 1` (as in
  /// Python). The same is true for negative values of `begin`.
  /// REQUIRES: `-(dims()+1) <= begin <= dims()`
  /// REQUIRES: `-(dims()+1) <= end <= dims()`
  void RemoveDimRange(int begin, int end);

  /// Same as `RemoveDimRange` but returns a `Status`.
  /// Use if unsure if requirements in `RemoveDimRange` are satistified, to
  /// prevent `CHECK`-fail crashes.
  Status RemoveDimRangeWithStatus(int begin, int end);

  /// Return whether the rank is unknown
  bool unknown_rank() const ;

  /// Return the number of dimensions in the tensor.
  /// Can be -1 meaning unknown rank for PartialTensorShape.
  int dims() const ;

  /// \brief Returns the number of elements in dimension `d`.
  /// REQUIRES: `0 <= d < dims()`
  // TODO(touts): Rename to `dimension()` to match
  // `Eigen::Tensor::dimension()`?
  int64_t dim_size(int d) const;

  /// Return true iff the rank and all of the dimensions are well defined
  // TODO(irving): Rename to is_fully_defined now that it's fast.
  bool IsFullyDefined() const ;

  /// For error messages.
  std::string DebugString() const;

  /// For iterating through the dimensions.
  TensorShapeIter<TensorShape> begin() const;
  TensorShapeIter<TensorShape> end() const;

  /// Returns true if `*this` and `b` have the same sizes. Ignores
  /// dimension names.
  bool IsSameSize(const TensorShape& b) const;
  bool operator==(const TensorShape& b) const { return IsSameSize(b); }
  bool operator!=(const TensorShape& b) const { return !IsSameSize(b); }

 private:
  // For access to TensorShapeBase(DataType).
  friend class Tensor;
};

/// Outputs `TensorShapeBase` to `std::ostream`.
inline std::ostream& operator<<(std::ostream& os, const TensorShape& ts) {
  return os << ts.DebugString();
}

/// Represents the value of one dimension in a TensorShape.
struct TensorShapeDim {
  explicit TensorShapeDim(int64_t s) : size(s) {}
  int64_t size;
};

// START_SKIP_DOXYGEN
template <class Shape>
class TensorShapeIter {
 public:
  TensorShapeIter(const Shape* shape, int d) : shape_(shape), d_(d) {}
  bool operator==(const TensorShapeIter& rhs) {
    DCHECK(shape_ == rhs.shape_);
    return d_ == rhs.d_;
  }
  bool operator!=(const TensorShapeIter& rhs) {
    DCHECK(shape_ == rhs.shape_);
    return d_ != rhs.d_;
  }
  void operator++() { ++d_; }
  TensorShapeDim operator*() { return TensorShapeDim(shape_->dim_size(d_)); }

 private:
  const Shape* shape_;
  int d_;
};
// END_SKIP_DOXYGEN

/// \brief Static helper routines for `TensorShape`. Includes a few common
/// predicates on a tensor shape.
class TensorShapeUtils {
 public:
  static bool IsScalar(const TensorShape& shape) { return shape.dims() == 0; }

  static bool IsVector(const TensorShape& shape) { return shape.dims() == 1; }

  static bool IsVectorOrHigher(const TensorShape& shape) {
    return shape.dims() >= 1;
  }

  static bool IsMatrix(const TensorShape& shape) { return shape.dims() == 2; }

  static bool IsSquareMatrix(const TensorShape& shape) {
    return shape.dims() == 2 && shape.dim_size(0) == shape.dim_size(1);
  }

  static bool IsMatrixOrHigher(const TensorShape& shape) {
    return shape.dims() >= 2;
  }

  /// \brief Returns a `TensorShape` whose dimensions are
  /// `dims[0]`, `dims[1]`, ..., `dims[n-1]`.
  static Status MakeShape(const int32_t* dims, int64_t n, TensorShape* out);
  static Status MakeShape(const int64_t* dims, int64_t n, TensorShape* out);


  /// \brief Returns true iff `shape` starts with `prefix`.
  static bool StartsWith(const TensorShape& shape, const TensorShape& prefix);

  /// \brief Returns true iff `shape` ends with `suffix`.
  static bool EndsWith(const TensorShape& shape, const TensorShape& suffix);

};


}  // namespace tensorflow
}  // namespace tensorflow_cpy
// clang-format on


#endif  // TENSORFLOW_CPY_CORE_FRAMEWORK_TENSOR_SHAPE_H_
