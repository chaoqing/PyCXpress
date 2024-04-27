#ifndef __PYCXPRESS_HPP__
#define __PYCXPRESS_HPP__

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "utils.hpp"

#if !defined(PYCXPRESS_EXPORT)
#if defined(WIN32) || defined(_WIN32)
#define PYCXPRESS_EXPORT __declspec(dllexport)
#else
#define PYCXPRESS_EXPORT __attribute__((visibility("default")))
#endif
#endif

namespace PyCXpress {
namespace py = pybind11;
using namespace utils;

typedef std::pair<void *, size_t> BufferPtr;
class PYCXPRESS_EXPORT            Buffer {
    typedef unsigned char Bytes;

    template <typename T>
    static py::array __to_array(const std::vector<size_t> &shape, void *data,
                                           size_t &max_size) {
                   std::vector<size_t> stride(shape.size());
                   *stride.rbegin() = sizeof(T);
                   auto ps          = shape.rbegin();
                   for (auto pt = stride.rbegin() + 1; pt != stride.rend(); pt++, ps++) {
                       *pt = *(pt - 1) * (*ps);
        }
                   auto real_size = stride.front() * shape.front();
                   if (max_size < real_size) {
                       throw std::runtime_error("Buffer size is too small");
        }
                   max_size = real_size;
                   return py::array_t<T>{shape, std::move(stride), (T *)(data),
                                         py::none()};
    }

public:
    Buffer() : m_size(0), m_data(nullptr), m_converter(nullptr) {}
    Buffer(size_t size, const std::string &data_type) : m_size(size) {
                   m_data = new Bytes[m_size];

                   if (data_type == "bool") {
                       m_converter = __to_array<bool>;
                       m_itemsize  = sizeof(bool);
        } else if (data_type == "int8_t") {
                       m_converter = __to_array<int8_t>;
                       m_itemsize  = sizeof(int8_t);
        } else if (data_type == "int16_t") {
                       m_converter = __to_array<int16_t>;
                       m_itemsize  = sizeof(int16_t);
        } else if (data_type == "int32_t") {
                       m_converter = __to_array<int32_t>;
                       m_itemsize  = sizeof(int32_t);
        } else if (data_type == "int64_t") {
                       m_converter = __to_array<int64_t>;
                       m_itemsize  = sizeof(int64_t);
        } else if (data_type == "uint8_t") {
                       m_converter = __to_array<uint8_t>;
                       m_itemsize  = sizeof(uint8_t);
        } else if (data_type == "uint16_t") {
                       m_converter = __to_array<uint16_t>;
                       m_itemsize  = sizeof(uint16_t);
        } else if (data_type == "uint32_t") {
                       m_converter = __to_array<uint32_t>;
                       m_itemsize  = sizeof(uint32_t);
        } else if (data_type == "uint64_t") {
                       m_converter = __to_array<uint64_t>;
                       m_itemsize  = sizeof(uint64_t);
        } else if (data_type == "float") {
                       m_converter = __to_array<float>;
                       m_itemsize  = sizeof(float);
        } else if (data_type == "double") {
                       m_converter = __to_array<double>;
                       m_itemsize  = sizeof(double);
        } else if (data_type == "char") {
                       m_converter = __to_array<char>;
                       m_itemsize  = sizeof(char);
        } else {
                       throw NotImplementedError(data_type);
        }
    }
    Buffer(Buffer &&ohs)
        : m_size(ohs.m_size),
          m_itemsize(ohs.m_itemsize),
          m_data(ohs.m_data),
          m_converter(ohs.m_converter) {
                   ohs.m_data = nullptr;
    }

    ~Buffer() {
                   delete[] m_data;
                   m_data = nullptr;
    }

    BufferPtr set(const std::vector<size_t> &shape) {
                   auto real_size = m_size;
                   m_array        = m_converter(shape, m_data, real_size);
                   return std::make_pair(m_data, real_size);
    }

    inline size_t itemsize() const { return m_itemsize; }

    py::array &array() { return m_array; }

    void reset() {
                   m_array = m_converter({m_size / m_itemsize}, m_data, m_size);
    }

private:
    size_t    m_size;
    size_t    m_itemsize;
    Bytes    *m_data;
    py::array m_array;
    py::array (*m_converter)(const std::vector<size_t> &, void *, size_t &);
};

class PYCXPRESS_EXPORT Model {
public:
    explicit Model(const std::string &path) {
        std::vector<char> module_name(path.data(), path.data() + path.length());
        if (module_name.empty() || module_name.back() == '.') {
            throw std::runtime_error("No model class provided");
        }
        auto iter = module_name.rbegin();
        while (iter + 1 != module_name.rend() && '.' != *iter) {
            ++iter;
        }
        if (iter + 1 == module_name.rend()) {
            throw std::runtime_error("not module provided");
        }
        auto ith             = std::distance(iter, module_name.rend());
        module_name[ith - 1] = 0;
        module_name.push_back('\0');
        initialize(module_name.data(), module_name.data() + ith);
    }

    Model(const Model &)            = delete;
    Model(Model &&)                 = delete;
    Model &operator=(const Model &) = delete;
    Model &operator=(Model &&)      = delete;

    ~Model() {
        m_buffers.clear();
        m_output_buffer_sizes.clear();
        m_model  = py::none();
        m_input  = py::none();
        m_output = py::none();
    }


    BufferPtr set_buffer(const std::string         &name,
                         const std::vector<size_t> &shape) {
        auto &buf  = m_buffers[name];
        auto  pBuf = buf.set(shape);
        m_input.attr("set_buffer_value")(name, buf.array());
        return pBuf;
    }

    std::pair<BufferPtr, std::vector<size_t>> get_buffer(
        const std::string &name) {
        auto &buf   = m_buffers[name];
        auto &array = buf.array();

        auto                iter = m_output_buffer_sizes.find(name);
        std::vector<size_t> shape;
        if (iter != m_output_buffer_sizes.end()) {
            shape = iter->second;
        } else {  // must be an input tensor, we do not suggest retrieve the
                  // data from this interface
            auto p = array.shape();
            shape.resize(array.ndim());
            std::copy_n(p, shape.size(), shape.begin());
        }

        auto nBytes = buf.itemsize() *
                      std::accumulate(shape.begin(), shape.end(), size_t(1),
                                      std::multiplies<size_t>());
        return std::make_pair(std::make_pair(array.request().ptr, nBytes),
                              std::move(shape));
    }

    void run() {
        m_model.attr("run")();

        auto get_buffer_shape = m_output.attr("get_buffer_shape");


        for (auto &kv : m_output_buffer_sizes) {
            kv.second.clear();
            py::tuple shape = get_buffer_shape(kv.first);

            for (auto &d : shape) {
                kv.second.push_back(d.cast<size_t>());
            }
        }
    }

private:
    void initialize(const char *module, const char *name) {
        m_model = py::module_::import(module).attr(name)();

        py::tuple spec;
        std::tie(m_input, m_output, spec) =
            m_model.attr("initialize")()
                .cast<std::tuple<py::object, py::object, py::tuple>>();

        auto set_buffer_value = m_output.attr("set_buffer_value");
        for (auto d = spec.begin(); d != spec.end(); d++) {
            auto       meta = d->cast<py::tuple>();
            const auto name = meta[0].cast<std::string>();
            auto       buf =
                m_buffers.insert({name, Buffer{meta[2].cast<size_t>(),
                                               meta[1].cast<std::string>()}});
            if (meta[3].cast<bool>()) {
                m_output_buffer_sizes[name] = {};
                auto &buffer                = buf.first->second;
                buffer.reset();
                set_buffer_value(name, buffer.array());
            }
        }
    }


    std::map<std::string, Buffer>              m_buffers;
    std::map<std::string, std::vector<size_t>> m_output_buffer_sizes;

    py::object m_model;
    py::object m_input;
    py::object m_output;
};

class PYCXPRESS_EXPORT PythonInterpreter {
public:
    explicit PythonInterpreter() {}

    PythonInterpreter(const PythonInterpreter &) = delete;
    PythonInterpreter(PythonInterpreter &&other) noexcept {
        other.is_valid = false;
    }
    PythonInterpreter &operator=(const PythonInterpreter &) = delete;
    PythonInterpreter &operator=(PythonInterpreter &&)      = delete;

    ~PythonInterpreter() { finalize(); }

    void initialize(bool init_signal_handlers = true, int argc = 0,
                    const char *const *argv                    = nullptr,
                    bool               add_program_dir_to_path = true) {
        // TODO: maybe explicitly `dlopen("/path/to/libpython3.x.so", RTLD_NOW |
        // RTLD_GLOBAL)` to avoid numpy import error
        py::initialize_interpreter(init_signal_handlers, argc, argv,
                                   add_program_dir_to_path);
        is_valid = true;
    }
    void finalize() {
        m_models.clear();

        if (is_valid) {
            py::finalize_interpreter();
            is_valid = false;
        }
    }

    Model &create_model(const std::string &path,
                        const std::string &name = "default") {
        if (!is_valid) {
            initialize();
        }
        if (m_models.find(name) == m_models.end()) {
            m_models[name] = std::make_unique<Model>(path);
        } else {
            std::cerr << "Warning: Model with name " << name
                      << " already exists" << std::endl;
        }
        return *m_models[name].get();
    }

private:
    bool                                          is_valid = false;
    std::map<std::string, std::unique_ptr<Model>> m_models;
};

};  // namespace PyCXpress

#endif  // __PYCXPRESS_HPP__
