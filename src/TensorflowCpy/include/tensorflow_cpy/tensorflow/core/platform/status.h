
#ifndef TENSORFLOW_CORE_PLATFORM_STATUS_H_
#define TENSORFLOW_CORE_PLATFORM_STATUS_H_

#include <iosfwd>
#include <string>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {

#if defined(__clang__)
// Only clang supports warn_unused_result as a type annotation.
class TF_MUST_USE_RESULT Status;
#endif

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

  tensorflow::error::Code code() const;

  const std::string& error_message() const ;

  bool operator==(const Status& x) const;
  bool operator!=(const Status& x) const;

  /// \brief If `ok()`, stores `new_status` into `*this`.  If `!ok()`,
  /// preserves the current status, but may augment with additional
  /// information about `new_status`.
  ///
  /// Convenient way of keeping track of the first error encountered.
  /// Instead of:
  ///   `if (overall_status.ok()) overall_status = new_status`
  /// Use:
  ///   `overall_status.Update(new_status);`
  void Update(const Status& new_status);

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


std::string error_name(error::Code code);

/*
inline std::string* TfCheckOpHelper(::tensorflow::Status v,
                                           const char* msg) {
  if (v.ok()) return nullptr;
  return &(v.error_message());
}

#define TF_DO_CHECK_OK(val, level)                                \
  while (auto _result = ::tensorflow::TfCheckOpHelper(val, #val)) \
  LOG(level) << *(_result)

#define TF_CHECK_OK(val) TF_DO_CHECK_OK(val, FATAL)
#define TF_QCHECK_OK(val) TF_DO_CHECK_OK(val, QFATAL)

// DEBUG only version of TF_CHECK_OK.  Compiler still parses 'val' even in opt
// mode.
#ifndef NDEBUG
#define TF_DCHECK_OK(val) TF_CHECK_OK(val)
#else
#define TF_DCHECK_OK(val) \
  while (false && (::tensorflow::OkStatus() == (val))) LOG(FATAL)
#endif
*/

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_STATUS_H_
