/// SavedModel loading functions and SavedModelBundle struct.

#ifndef TENSORFLOW_CPY_CC_SAVED_MODEL_LOADER_H_
#define TENSORFLOW_CPY_CC_SAVED_MODEL_LOADER_H_

#include <string>
#include <unordered_set>

#include <tensorflow/core/protobuf/meta_graph.pb.h>
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
  //const protobuf::Map<std::string, SignatureDef>& GetSignatures() const {
    //return meta_graph_def.signature_def();
  //}

  std::unique_ptr<Session> session;
  MetaGraphDef meta_graph_def;
  //std::unique_ptr<GraphDebugInfo> debug_info;
};


// Restore variable and resources in the SavedModel export dir for the
// indicated metagraph.
// The recommended way to load a saved model is to call LoadSavedModel,
// which provides an already initialized Metagraph, Session, and DebugInfo.
Status RestoreSession(const RunOptions& run_options,
                      const MetaGraphDef& meta_graph, const std::string& export_dir,
                      std::unique_ptr<Session>* session);

// Initialize a session which wraps this metagraph.
// The recommended way to load a saved model is to call LoadSavedModel,
// which provides an already initialized Metagraph, Session, and DebugInfo.
Status LoadMetagraphIntoSession(const SessionOptions& session_options,
                                const MetaGraphDef& meta_graph,
                                std::unique_ptr<Session>* session);

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

/// Checks whether the provided directory could contain a SavedModel. Note that
/// the method does not load any data by itself. If the method returns `false`,
/// the export directory definitely does not contain a SavedModel. If the method
/// returns `true`, the export directory may contain a SavedModel but provides
/// no guarantee that it can be loaded.
bool MaybeSavedModelDirectory(const std::string& export_dir);

}  // namespace tensorflow
}  // namespace tensorflow_cpy
// clang-format on


#endif  // TENSORFLOW_CPY_CC_SAVED_MODEL_LOADER_H_
