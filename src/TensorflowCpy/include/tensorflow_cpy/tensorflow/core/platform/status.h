
#ifndef TENSORFLOW_CPY_CORE_PLATFORM_STATUS_H_
#define TENSORFLOW_CPY_CORE_PLATFORM_STATUS_H_

#include <tensorflow/core/protobuf/error_codes.pb.h>

#include <iosfwd>
#include <string>

// clang-format off
namespace tensorflow_cpy {
namespace tensorflow {
  using namespace ::tensorflow;

namespace errors {

typedef ::tensorflow::error::Code Code;

}  // namespace errors
/// @ingroup core
/// Denotes success or failure of a call in Tensorflow.
class Status {
 public:
  /// Create a success status.
  Status() {}

  /// Copy the specified status.
  Status(const Status& s);
  Status& operator=(const Status& s);

  // Prefer using OkStatus().
  static Status OK() { return Status(); }

  /// Returns true iff the status indicates success.
  bool ok() const;

  errors::Code code() const;

  const std::string& error_message() const ;

  /// \brief Return a string representation of this status suitable for
  /// printing. Returns the string `"OK"` for success.
  ///
  /// By default, it returns combination of the error code name, the message and
  /// any associated payload messages. This string is designed simply to be
  /// human readable and its exact format should not be load bearing. Do not
  /// depend on the exact format of the result of `ToString()` which is subject
  /// to change.
  std::string ToString() const;

  // Ignores any errors. This method does nothing except potentially suppress
  // complaints from any tools that are checking that errors are not dropped on
  // the floor.
  void IgnoreError() const;

 private:
};

// OkStatus()
//
// Returns an OK status, equivalent to a default constructed instance. Prefer
// usage of `OkStatus()` when constructing such an OK status.
Status OkStatus();

/// @ingroup core
std::ostream& operator<<(std::ostream& os, const Status& x);

}  // namespace tensorflow
}  // namespace tensorflow_cpy
// clang-format on


#endif  // TENSORFLOW_CPY_CORE_PLATFORM_STATUS_H_
