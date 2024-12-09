#include <tensorflow_cpy/framework.h>

#include <filesystem>
#include <iostream>
#include <sstream>
namespace fs = std::filesystem;

#include <signal.h>
#include <stdio.h>
#include <sysexits.h>
#include <unistd.h>
void wait_for_debugger_attach() {
    std::cerr << "[info]: pid=" << (int)getpid() << ": wait for debugger attach..." << std::endl;
    if (raise(SIGSTOP) == -1) {
        perror("");
        exit(EX_OSERR);
    }
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wdeprecated-builtins"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Winconsistent-missing-override"

#pragma GCC diagnostic ignored "-Wswitch"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-variable"


#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>
#include <tensorflow/core/common_runtime/device/device_id.h>
#include <tensorflow/core/common_runtime/device/device_id_utils.h>
#include <tensorflow/core/common_runtime/device/device_mem_allocator.h>
#include <tensorflow/core/common_runtime/device_mgr.h>
#include <tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.h>
#include <tensorflow/core/common_runtime/gpu/gpu_init.h>
#include "tensorflow/core/platform/types.h"
#include <tensorflow/core/framework/tensor.h>
#include <tensorflow/core/framework/device.h>
#include <tensorflow/core/framework/op.h>
#include <tensorflow/core/framework/types.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/util/stream_executor_util.h>

#pragma GCC diagnostic pop

namespace tf = tensorflow;
namespace se = stream_executor;

class TensorBufferView : public tf::TensorBuffer {
    std::size_t m_len;

public:
    TensorBufferView(void* data, std::size_t len) : tf::TensorBuffer(data), m_len(len) {}

    std::size_t size() const override { return m_len; }

    tf::AllocatorMemoryType GetMemoryType() const override {
        return tf::AllocatorMemoryType::kDevice;
    }

    tf::TensorBuffer* root_buffer() override { return this; }
    bool              OwnsMemory() const override { return false; }

    void FillAllocationDescription(tf::AllocationDescription* proto) const override {
        proto->set_allocated_bytes(static_cast<int64_t>(m_len));
        proto->set_allocator_name("TensorBufferView");
        proto->set_ptr(reinterpret_cast<uintptr_t>(this->data()));
    }
};

// TensorBufferView can only be created with "new" and hosted inside a unique_ptr
using TensorBufferViewPtr = tf::core::RefCountPtr<TensorBufferView>;

int tensorflow_cpy::main_whole_flow(int argc, char** argv) {
    if (argc == 0) {
        return 1;
    }

    // configure session
    tf::SessionOptions sessionOption;
    tf::RunOptions     runOption;

    tf::ConfigProto& config = sessionOption.config;
    config.set_inter_op_parallelism_threads(true);
    config.set_intra_op_parallelism_threads(true);
    config.set_use_per_session_threads(false);
    (*config.mutable_device_count())["GPU"] = 1;

    tf::GPUOptions& gpuOption = *config.mutable_gpu_options();
    gpuOption.set_visible_device_list("");
    gpuOption.set_force_gpu_compatible(true);

    if (1 == 1) {  // VirtualDevices and MemoryFraction can not coexist
        auto& device = *gpuOption.mutable_experimental()->add_virtual_devices();
        device.add_memory_limit_mb(100.0);
        device.add_priority(0);
    } else {
        constexpr auto fraction = 0.7;
        if (fraction == 0) {
            gpuOption.set_allow_growth(true);
        } else {
            gpuOption.set_per_process_gpu_memory_fraction(fraction);
        }
    }

    // Setting cuda device flags
    // cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

    // load ops
    tf::Env* env = tf::Env::Default();

    assert(tf::OpRegistry::Global()->ProcessRegistrations().ok());
    tf::OpRegistry::Global()->DeferRegistrations();

    const fs::path ops_dir{"./sample/models/ops/"};
    if (fs::is_directory(ops_dir)) {
        for (const auto& op_so : fs::directory_iterator{ops_dir}) {
            if (op_so.is_directory()) continue;
            void* opHandle = nullptr;
            assert(env->LoadDynamicLibrary(op_so.path().c_str(), &opHandle).ok());
        }
    }
    assert(tf::OpRegistry::Global()->ProcessRegistrations().ok());

    // load model
    tf::SavedModelBundle model;
    std::string          model_path(argv[1]);
    auto status = tf::LoadSavedModel(sessionOption, runOption, model_path, {"serve"}, &model);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
    }
    assert(status.ok());

    // get devices
    const tf::DeviceMgr* deviceMgr = nullptr;
    assert(model.GetSession()->LocalDeviceManager(&deviceMgr).ok());
    tf::Device* cpuDevice = nullptr;
    for (auto d : deviceMgr->ListDevices()) {
        if (d->device_type() == "CPU") {
            cpuDevice = d;
            break;
        }
    }
    std::vector<tensorflow::DeviceAttributes> devices;
    assert(model.GetSession()->ListDevices(&devices).ok());
    auto iter = std::find_if(devices.begin(), devices.end(),
                             [](const auto& d) { return d.device_type() == "GPU"; });
    assert(iter != devices.end());
    std::string gpuDeviceName(iter->name());
    // wait_for_debugger_attach();

    // create tensor for memory buffer
    tf::PlatformDeviceId deviceId(0);
    se::Platform*        platform = tf::GPUMachineManager();
    se::StreamExecutor*  streamExecutor
        = tf::DeviceIdUtil::ExecutorForPlatformDeviceId(platform, deviceId).ValueOrDie();
    assert(streamExecutor != nullptr);

    auto CreateGPUAllocator = [&streamExecutor, &deviceId](size_t size) {
        constexpr auto alignment       = 512;
        auto           gpuSubAllocator = std::make_unique<tf::DeviceMemAllocator>(
            streamExecutor, deviceId, false, std::vector<tf::SubAllocator::Visitor>{},
            std::vector<tf::SubAllocator::Visitor>{});
        std::ostringstream name;
        name << "GPU_" << deviceId.value() << "_bfc";
        auto gpuAllocator = std::make_shared<tf::GPUBFCAllocator>(
            std::move(gpuSubAllocator), ((size + alignment - 1) / alignment) * alignment,
            name.str(), tf::GPUBFCAllocator::Options{});

        return gpuAllocator;
    };

    // create buffers
    tf::TensorShape shape({tf::int64(3)});
    shape.AddDim(tf::int64(4));
    assert(shape.dims() == 2);
    assert(shape.dim_size(1) == 4);
    assert(shape.num_elements() == 12);

    // cpu buffer
    tf::AllocatorAttributes hostAllocatorAttr;
    hostAllocatorAttr.set_on_host(true);
    hostAllocatorAttr.set_gpu_compatible(true);

    auto       cpuAllocator = cpuDevice->GetAllocator(hostAllocatorAttr);
    tf::Tensor hostTensor(cpuAllocator, tf::DataTypeToEnum<uint8_t>::value, shape);
    if (hostTensor.GetMemoryType() != tf::AllocatorMemoryType::kDevice) {
        std::fill_n((uint8_t*)hostTensor.data(), shape.num_elements(), 1);
        std::cout << hostTensor.DebugString() << std::endl;
    }
    // Maybe let cuda know this is hostPinned memory so that AsyncMemoryCopy works
    // cudaHostRegister(hostTensor.data(), hostTensor.TotalBytes(), cudaHostRegisterPortable);


    // cudaStream_t cudaStream;
    // int lowP, highP;
    // cydaDeviceGetStreamPriorityRange(&lowP, &highP);
    // cudaStreamCreateWithPriority(&cudaStream, cudaStreamNonBlocking, lowP);
    // cudaStreamDestory(cudaStream);

    // gpu buffer
    constexpr auto INPUT_DATA_TYPE   = tf::DataTypeToEnum<uint8_t>::value;
    const auto     inputDataTypeSize = tf::DataTypeSize(INPUT_DATA_TYPE);
    static_assert(std::is_same_v<uint8_t, typename tf::EnumToDataType<INPUT_DATA_TYPE>::Type>);
    constexpr auto DEVICE_MEMORY_ALIGNMENT = 512;
    auto           gpuAllocator            = CreateGPUAllocator(
        /*A*/ ((shape.num_elements() * 10 * inputDataTypeSize) + DEVICE_MEMORY_ALIGNMENT - 1)
            / DEVICE_MEMORY_ALIGNMENT * DEVICE_MEMORY_ALIGNMENT
        + /*B*/ ((shape.num_elements() * 10 * inputDataTypeSize) + DEVICE_MEMORY_ALIGNMENT - 1)
              / DEVICE_MEMORY_ALIGNMENT * DEVICE_MEMORY_ALIGNMENT
        + /*C*/ ((shape.num_elements() * 10 * inputDataTypeSize) + DEVICE_MEMORY_ALIGNMENT - 1)
              / DEVICE_MEMORY_ALIGNMENT * DEVICE_MEMORY_ALIGNMENT);
    std::cerr << "Fresh GPU ======== \n" << gpuAllocator->GetStats()->DebugString() << std::endl;
    tf::Tensor input_tensor_A(gpuAllocator.get(), INPUT_DATA_TYPE, shape);
    std::cerr << "After A GPU ======== \n" << gpuAllocator->GetStats()->DebugString() << std::endl;
    tf::Tensor input_tensor_B(gpuAllocator.get(), INPUT_DATA_TYPE, shape);
    std::cerr << "After B GPU ======== \n" << gpuAllocator->GetStats()->DebugString() << std::endl;

    int64_t totalMem = 0, freeMem = 0;
    assert(streamExecutor->DeviceMemoryUsage(&freeMem, &totalMem));
    std::cerr << "stream " << streamExecutor->GetDeviceDescription().pci_bus_id() << " have "
              << freeMem / 1024. / 1024 << " out of " << totalMem / 1024. / 1024 << " MB memory"
              << std::endl;


    // model to function handler
    tf::Session::CallableHandle modelFuncHandle;
    tf::CallableOptions         callOptions;

    callOptions.add_feed("import/a:0");
    callOptions.add_feed("import/b:0");
    callOptions.add_fetch("import/add:0");
    callOptions.clear_fetch_devices();
    callOptions.mutable_feed_devices()->insert({"import/a:0", gpuDeviceName});
    callOptions.mutable_feed_devices()->insert({"import/b:0", gpuDeviceName});
    callOptions.set_fetch_skip_sync(true);

    status = model.GetSession()->MakeCallable(callOptions, &modelFuncHandle);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
    }
    assert(status.ok());

    // create tensor from memory buffer
    std::vector<tf::Tensor> inputs;
    TensorBufferViewPtr     bufferViewA(new TensorBufferView(input_tensor_A.data(), 6));
    inputs.emplace_back(tf::DataType{tf::DT_UINT8}, tf::TensorShape{2, 3},
                        bufferViewA->root_buffer());
    TensorBufferViewPtr bufferViewB(new TensorBufferView(input_tensor_B.data(), 6));
    inputs.emplace_back(tf::DataType{tf::DT_UINT8}, tf::TensorShape{2, 3},
                        bufferViewB->root_buffer());
    // std::cerr<<bufferViewA->RefCount()<<" "<<bufferViewB->RefCount()<<std::endl;

    if (input_tensor_A.GetMemoryType() == tf::AllocatorMemoryType::kDevice) {
        auto ptr = tf::StreamExecutorUtil::AsDeviceMemory<uint8_t>(input_tensor_A);
        // a[*] == 5;
        assert(streamExecutor
                   ->SynchronousMemSet(&ptr, 5, input_tensor_A.NumElements() * sizeof(uint8_t))
                   .ok());
    }
    // std::cerr<<input_tensor_A.IsInitialized()<<std::endl;
    std::vector<uint8_t> input_tensor_B_host(shape.num_elements());
    // b[i] == i;
    for (auto i = 0; i < shape.num_elements(); i++) {
        input_tensor_B_host[i] = i;
    }
    se::DeviceMemoryBase ptr{input_tensor_B.data(), input_tensor_B.NumElements() * sizeof(uint8_t)};
    assert(streamExecutor
               ->SynchronousMemcpyH2D(input_tensor_B_host.data(),
                                      input_tensor_B_host.size() * sizeof(uint8_t), &ptr)
               .ok());
    // assert(streamExecutor->SynchronizeAllActivity());

    // run the model
    std::vector<tf::Tensor> outputs;
    status = model.GetSession()->RunCallable(modelFuncHandle, inputs, &outputs, nullptr);
    if (!status.ok()) {
        std::cerr << status.ToString() << std::endl;
    }
    assert(status.ok());
    const auto&          result = outputs.front();
    se::DeviceMemoryBase res_ptr{result.data(), result.NumElements() * sizeof(uint8_t)};
    uint8_t*             hostResPtr = (uint8_t*)hostTensor.data();
    assert(streamExecutor
               ->SynchronousMemcpyD2H(res_ptr, result.NumElements() * sizeof(uint8_t), hostResPtr)
               .ok());
    std::cout << "result shape: " << result.dims() << "(" << result.dim_size(0) << ", "
              << result.dim_size(1) << ")" << std::endl;
    for (long i = 0; i < result.NumElements(); i++) {
        std::cout << "  " << i << " -> " << int(hostResPtr[i]) << std::endl;
    }

    assert(model.GetSession()->ReleaseCallable(modelFuncHandle).ok());

    std::cerr << "completed" << std::endl;

    return 0;
}
