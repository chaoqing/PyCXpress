#include <tensorflow_cpy/tensorflow.h>

namespace tensorflow_cpy {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
    namespace stream_executor {
        bool StreamExecutor::DeviceMemoryUsage(long*, long*) const { return true; }
        const DeviceDescription& StreamExecutor::GetDeviceDescription() const {
            return DeviceDescription::CreateDummy();
        }
        DeviceDescription::DeviceDescription() {}
        void         StreamExecutor::Deallocate(DeviceMemoryBase*) {}
        port::Status StreamExecutor::SynchronousMemcpyD2H(DeviceMemoryBase const&, long, void*) {
            return port::Status::OK();
        }
        port::Status StreamExecutor::SynchronousMemcpyH2D(void const*, long, DeviceMemoryBase*) {
            return port::Status::OK();
        }
        port::Status StreamExecutor::SynchronousMemSet(DeviceMemoryBase*, int, unsigned long) {
            return port::Status::OK();
        }
        void* StreamExecutor::UnifiedMemoryAllocate(unsigned long) { return nullptr; }
        void  StreamExecutor::UnifiedMemoryDeallocate(void*) {}
    };  // namespace stream_executor
    namespace tensorflow {
        namespace se = stream_executor;

        SessionOptions::SessionOptions() {}
        namespace internal {
            LogMessageFatal::LogMessageFatal(const char* file, int line)
                : LogMessage(file, line, FATAL) {}
            LogMessageFatal::~LogMessageFatal() { exit(1); }
            LogMessage::LogMessage(const char* file, int line, int severity) {}
            LogMessage::~LogMessage() {}
        };  // namespace internal

        namespace core {
            RefCounted::RefCounted() {}
            bool RefCounted::Unref() const { return true; }
        }  // namespace core


        Env*        Env::Default() { return nullptr; }
        OpRegistry* OpRegistry::Global() { return nullptr; }
        Status      OpRegistry::ProcessRegistrations() const { return Status::OK(); }
        void        OpRegistry::DeferRegistrations() {}

        std::string AllocatorStats::DebugString() const { return ""; }
        int         DataTypeSize(DataType) { return 0; }


        Allocator::~Allocator() {}
        GPUBFCAllocator::GPUBFCAllocator(
            std::unique_ptr<SubAllocator, std::default_delete<SubAllocator> >, unsigned long,
            std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > const&,
            GPUBFCAllocator::Options const&) {}

        std::string GPUBFCAllocator::Name() { return ""; }
        void* GPUBFCAllocator::AllocateRaw(size_t alignment, size_t num_bytes) { return nullptr; }
        void  GPUBFCAllocator::DeallocateRaw(void* ptr) {}


        Status        ValidateGPUMachineManager() { return Status::OK(); }
        se::Platform* GPUMachineManager() { return nullptr; }
        std::string   GpuPlatformName() { return ""; }
        Status LoadSavedModel(const SessionOptions& session_options, const RunOptions& run_options,
                              const std::string&                     export_dir,
                              const std::unordered_set<std::string>& tags,
                              SavedModelBundle* const                bundle) {
            return Status::OK();
        }
        errors::Code Status::code() const { return errors::Code::OK; }

        const std::string& Status::error_message() const {
            static std::string null = "";
            return null;
        }

        void        Status::Update(const Status& new_status) {}
        void        Status::IgnoreError() const {}
        bool        Status::ok() const { return true; }
        Status&     Status::operator=(Status const&) { return *this; }
        std::string Status::ToString() const { return ""; }
        SubAllocator::SubAllocator(
            std::vector<std::function<void(void*, int, unsigned long)>,
                        std::allocator<std::function<void(void*, int, unsigned long)> > > const&,
            std::vector<std::function<void(void*, int, unsigned long)>,
                        std::allocator<std::function<void(void*, int, unsigned long)> > > const&) {}
        void SubAllocator::VisitAlloc(void*, int, unsigned long) {}
        void SubAllocator::VisitFree(void*, int, unsigned long) {}

        TensorShape::TensorShape() {}
        void    TensorShape::AddDim(long) {}
        int     TensorShape::dims() const { return 0; }
        int64_t TensorShape::dim_size(int) const { return 0; }
        int64_t TensorShape::num_elements() const { return 0; }
        TensorShape::TensorShape(std::initializer_list<long>) {}

        std::string Tensor::DebugString(int) const { return ""; }
        Tensor::~Tensor() {}
        Tensor::Tensor(Allocator*, tensorflow::DataType, TensorShape const&) {}
        Tensor::Tensor(Tensor const&) {}
        Tensor::Tensor(DataType, TensorShape const&, TensorBuffer*) {}
        size_t Tensor::TotalBytes() const { return 0; }
    };  // namespace tensorflow
#pragma GCC diagnostic pop
};  // namespace tensorflow_cpy
