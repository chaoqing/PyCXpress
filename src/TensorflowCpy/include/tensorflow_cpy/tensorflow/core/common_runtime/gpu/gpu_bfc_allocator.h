
#ifndef TENSORFLOW_CPY_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
#define TENSORFLOW_CPY_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_

#include <memory>
#include <string>

#include "../../../core/framework/allocator.h"

// clang-format off
namespace tensorflow_cpy {
namespace tensorflow {
  using namespace ::tensorflow;

// A GPU memory allocator that implements a 'best-fit with coalescing'
// algorithm.
class GPUBFCAllocator : public Allocator {
 public:
  // See BFCAllocator::Options.
  struct Options {
    // Overridden by TF_FORCE_GPU_ALLOW_GROWTH if that envvar is set.
    bool allow_growth = false;


    double fragmentation_fraction = 0;
    bool allow_retry_on_failure = true;
  };

  GPUBFCAllocator(std::unique_ptr<SubAllocator> sub_allocator,
                  size_t total_memory, const std::string& name, const Options& opts);

  ~GPUBFCAllocator() override {}

  std::string Name() override;
  void* AllocateRaw(size_t alignment, size_t num_bytes) override;
  void DeallocateRaw(void* ptr) override;

  TF_DISALLOW_COPY_AND_ASSIGN(GPUBFCAllocator);
};

}  // namespace tensorflow
}  // namespace tensorflow_cpy
// clang-format on


#endif  // TENSORFLOW_CPY_CORE_COMMON_RUNTIME_GPU_GPU_BFC_ALLOCATOR_H_
