#pragma once

#include "./tensorflow/stream_executor/device_memory.h"
#include "./tensorflow/stream_executor/stream_executor_pimpl.h"
#include "./tensorflow/stream_executor/platform.h"

#include "./tensorflow/cc/saved_model/loader.h"
#include "./tensorflow/cc/saved_model/tag_constants.h"

#include "./tensorflow/core/util/stream_executor_util.h"

#include "./tensorflow/core/platform/logging.h"
#include "./tensorflow/core/platform/statusor.h"
#include "./tensorflow/core/platform/refcount.h"
#include "./tensorflow/core/platform/types.h"
#include "./tensorflow/core/platform/macros.h"
#include "./tensorflow/core/platform/status.h"
#include "./tensorflow/core/platform/env.h"

#include "./tensorflow/core/public/session_options.h"
#include "./tensorflow/core/public/session.h"

#include "./tensorflow/core/common_runtime/device_mgr.h"
#include "./tensorflow/core/common_runtime/gpu/gpu_init.h"
#include "./tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h"
#include "./tensorflow/core/common_runtime/device/device_id_utils.h"
#include "./tensorflow/core/common_runtime/device/device_mem_allocator.h"
#include "./tensorflow/core/common_runtime/device/device_id.h"

#include "./tensorflow/core/framework/allocator.h"
#include "./tensorflow/core/framework/types.h"
#include "./tensorflow/core/framework/tensor_shape.h"
#include "./tensorflow/core/framework/tensor.h"
#include "./tensorflow/core/framework/device.h"
#include "./tensorflow/core/framework/op.h"

namespace tensorflow_cpy {
namespace tensorflow {
  using namespace ::tensorflow;
}  // namespace tensorflow
}  // namespace tensorflow_cpy
