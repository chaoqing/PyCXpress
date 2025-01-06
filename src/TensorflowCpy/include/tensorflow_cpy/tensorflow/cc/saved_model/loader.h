/// SavedModel loading functions and SavedModelBundle struct.

#ifndef TENSORFLOW_CPY_CC_SAVED_MODEL_LOADER_H_
#define TENSORFLOW_CPY_CC_SAVED_MODEL_LOADER_H_

#include <tensorflow/core/protobuf/meta_graph.pb.h>

#include <string>
#include <unordered_set>

#include "../../core/platform/status.h"
#include "../../core/public/session.h"

// clang-format off
namespace tensorflow_cpy {
namespace tensorflow {
  using namespace ::tensorflow;


/// SavedModel representation once the SavedModel is loaded from storage.
///
/// NOTE: Prefer to use SavedModelBundleLite in new code, as it consumes less
/// RAM.
struct SavedModelBundle {
  /// A TensorFlow Session does not Close itself on destruction. To avoid
  /// resource leaks, we explicitly call Close on Sessions that we create.
  ~SavedModelBundle() {
    if (session) {
      session->Close().IgnoreError();
    }
  }

  SavedModelBundle() = default;

  Session* GetSession() const { return session.get(); }

  std::unique_ptr<Session> session;
  MetaGraphDef meta_graph_def;
  //std::unique_ptr<GraphDebugInfo> debug_info;
};


/// Loads a SavedModel from the specified export directory. The MetaGraphDef
/// to be loaded is identified by the supplied tags, corresponding exactly to
/// the set of tags used at SavedModel build time. Stores a SavedModel bundle in
/// *bundle with a session and the requested MetaGraphDef, if found.
///
/// NOTE: Prefer the overload that takes a SavedModelBundleLite* in new code.
Status LoadSavedModel(const SessionOptions& session_options,
                      const RunOptions& run_options, const std::string& export_dir,
                      const std::unordered_set<std::string>& tags,
                      SavedModelBundle* const bundle);
}  // namespace tensorflow
}  // namespace tensorflow_cpy
// clang-format on


#endif  // TENSORFLOW_CPY_CC_SAVED_MODEL_LOADER_H_
