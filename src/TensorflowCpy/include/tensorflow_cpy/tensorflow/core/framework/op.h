#ifndef TENSORFLOW_CPY_CORE_FRAMEWORK_OP_H_
#define TENSORFLOW_CPY_CORE_FRAMEWORK_OP_H_

#include "../../core/platform/status.h"

// clang-format off
namespace tensorflow_cpy {
namespace tensorflow {
  using namespace ::tensorflow;

// The standard implementation of OpRegistryInterface, along with a
// global singleton used for registering ops via the REGISTER
// macros below.  Thread-safe.
//
// Example registration:
//   OpRegistry::Global()->Register(
//     [](OpRegistrationData* op_reg_data)->Status {
//       // Populate *op_reg_data here.
//       return Status::OK();
//   });
class OpRegistry {
 public:
  //typedef std::function<Status(OpRegistrationData*)> OpRegistrationDataFactory;

  OpRegistry();
  ~OpRegistry();

  // Returns ASCII-format OpList for all registered OpDefs (except
  // those with names starting with '_' if include_internal == false).
  std::string DebugString(bool include_internal) const;

  // A singleton available at startup.
  static OpRegistry* Global();

  // Process the current list of deferred registrations. Note that calls to
  // Export, LookUp and DebugString would also implicitly process the deferred
  // registrations. Returns the status of the first failed op registration or
  // Status::OK() otherwise.
  Status ProcessRegistrations() const;

  // Defer the registrations until a later call to a function that processes
  // deferred registrations are made. Normally, registrations that happen after
  // calls to Export, LookUp, ProcessRegistrations and DebugString are processed
  // immediately. Call this to defer future registrations.
  void DeferRegistrations();

  // Clear the registrations that have been deferred.
  void ClearDeferredRegistrations();

 private:
};

}  // namespace tensorflow
}  // namespace tensorflow_cpy
// clang-format on


#endif  // TENSORFLOW_CPY_CORE_FRAMEWORK_OP_H_
