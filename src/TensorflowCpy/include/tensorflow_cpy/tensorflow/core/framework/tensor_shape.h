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
  std::vector<int64_t> m_dims;
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

  /// \brief Add a dimension to the end ("inner-most").
  /// REQUIRES: `size >= 0`
  void AddDim(int64_t size);

  /// Appends all the dimensions from `shape`.
  void AppendShape(const TensorShape& shape);

  /// \brief Insert a dimension somewhere in the `TensorShape`.
  /// REQUIRES: `0 <= d <= dims()`
  /// REQUIRES: `size >= 0`
  void InsertDim(int d, int64_t size);

  /// \brief Modifies the size of the dimension `d` to be `size`
  /// REQUIRES: `0 <= d < dims()`
  /// REQUIRES: `size >= 0`
  void set_dim(int d, int64_t size);

  /// \brief Removes last `n` dimensions from the `TensorShape`.
  /// REQUIRES: `0 <= n <= dims()`
  void RemoveLastDims(int n) {
    CHECK_LE(n, dims());
    RemoveDimRange(dims() - n, dims());
  }

  /// \brief Removes the dimensions in range `[begin:end)` from `TensorShape`.
  /// Negative values of `end` are interpreted as `dims() + end + 1` (as in
  /// Python). The same is true for negative values of `begin`.
  /// REQUIRES: `-(dims()+1) <= begin <= dims()`
  /// REQUIRES: `-(dims()+1) <= end <= dims()`
  void RemoveDimRange(int begin, int end);

  /// Return the number of dimensions in the tensor.
  /// Can be -1 meaning unknown rank for PartialTensorShape.
  int dims() const ;

  /// \brief Returns the number of elements in dimension `d`.
  /// REQUIRES: `0 <= d < dims()`
  int64_t dim_size(int d) const;

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
