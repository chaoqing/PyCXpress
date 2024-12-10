#ifndef TENSORFLOW_CPY_CC_SAVED_MODEL_TAG_CONSTANTS_H_
#define TENSORFLOW_CPY_CC_SAVED_MODEL_TAG_CONSTANTS_H_

// clang-format off
namespace tensorflow_cpy {
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
}  // namespace tensorflow_cpy
// clang-format on


#endif  // TENSORFLOW_CPY_CC_SAVED_MODEL_TAG_CONSTANTS_H_
