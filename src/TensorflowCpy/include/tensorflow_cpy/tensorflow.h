#ifndef TENSORFLOW_CPY_TENSORFLOW_H_
#define TENSORFLOW_CPY_TENSORFLOW_H_

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-warning-option"

#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wdeprecated-builtins"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Winconsistent-missing-override"

#pragma GCC diagnostic ignored "-Wswitch"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-private-field"

#include "./tensorflow/cc/saved_model/loader.h"                            // IWYU pragma: keep
#include "./tensorflow/cc/saved_model/tag_constants.h"                     // IWYU pragma: keep
#include "./tensorflow/core/common_runtime/device/device_id.h"             // IWYU pragma: keep
#include "./tensorflow/core/common_runtime/device/device_id_utils.h"       // IWYU pragma: keep
#include "./tensorflow/core/common_runtime/device/device_mem_allocator.h"  // IWYU pragma: keep
#include "./tensorflow/core/common_runtime/device_mgr.h"                   // IWYU pragma: keep
#include "./tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"        // IWYU pragma: keep
#include "./tensorflow/core/common_runtime/gpu/gpu_init.h"                 // IWYU pragma: keep
#include "./tensorflow/core/framework/allocator.h"                         // IWYU pragma: keep
#include "./tensorflow/core/framework/device.h"                            // IWYU pragma: keep
#include "./tensorflow/core/framework/op.h"                                // IWYU pragma: keep
#include "./tensorflow/core/framework/tensor.h"                            // IWYU pragma: keep
#include "./tensorflow/core/framework/tensor_shape.h"                      // IWYU pragma: keep
#include "./tensorflow/core/framework/types.h"                             // IWYU pragma: keep
#include "./tensorflow/core/platform/env.h"                                // IWYU pragma: keep
#include "./tensorflow/core/platform/logging.h"                            // IWYU pragma: keep
#include "./tensorflow/core/platform/macros.h"                             // IWYU pragma: keep
#include "./tensorflow/core/platform/refcount.h"                           // IWYU pragma: keep
#include "./tensorflow/core/platform/status.h"                             // IWYU pragma: keep
#include "./tensorflow/core/platform/statusor.h"                           // IWYU pragma: keep
#include "./tensorflow/core/platform/types.h"                              // IWYU pragma: keep
#include "./tensorflow/core/public/session.h"                              // IWYU pragma: keep
#include "./tensorflow/core/public/session_options.h"                      // IWYU pragma: keep
#include "./tensorflow/core/util/stream_executor_util.h"                   // IWYU pragma: keep
#include "./tensorflow/stream_executor/device_memory.h"                    // IWYU pragma: keep
#include "./tensorflow/stream_executor/platform.h"                         // IWYU pragma: keep
#include "./tensorflow/stream_executor/stream_executor_pimpl.h"            // IWYU pragma: keep
#pragma GCC diagnostic pop

namespace tensorflow_cpy {
    namespace tensorflow {
        using namespace ::tensorflow;
    }  // namespace tensorflow
}  // namespace tensorflow_cpy
#endif  // TENSORFLOW_CPY_TENSORFLOW_H_
