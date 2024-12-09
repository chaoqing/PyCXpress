#ifndef TENSORFLOW_CC_SAVED_MODEL_TAG_CONSTANTS_H_
#define TENSORFLOW_CC_SAVED_MODEL_TAG_CONSTANTS_H_

namespace tensorflow {

/// Tag for the `gpu` graph.
constexpr char kSavedModelTagGpu[] = "gpu";

/// Tag for the `tpu` graph.
constexpr char kSavedModelTagTpu[] = "tpu";

/// Tag for the `serving` graph.
constexpr char kSavedModelTagServe[] = "serve";

/// Tag for the `training` graph.
constexpr char kSavedModelTagTrain[] = "train";

}  // namespace tensorflow

#endif  // TENSORFLOW_CC_SAVED_MODEL_TAG_CONSTANTS_H_
