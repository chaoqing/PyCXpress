#ifndef __PYCXPRESS_HPP__
#define __PYCXPRESS_HPP__

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "Utils.hpp"

namespace PyCXpress {
namespace py = pybind11;
using namespace utils;

class Buffer {
    typedef unsigned char Bytes;

    template <typename T>
    static py::array __to_array(const std::vector<size_t> &shape, void *data) {
        std::vector<size_t> stride(shape.size());
        *stride.rbegin() = sizeof(T);
        auto ps          = shape.rbegin();
        for (auto pt = stride.rbegin() + 1; pt != stride.rend(); pt++, ps++) {
            *pt = *(pt - 1) * (*ps);
        }
        return py::array_t<T>{shape, std::move(stride), (T *)(data),
                              py::none()};
    }

public:
    Buffer() : m_size(0), m_data(nullptr), m_converter(nullptr) {}
    Buffer(size_t size, const std::string &data_type) : m_size(size) {
        m_data   = new Bytes[m_size];
        m_length = size;

        if (data_type == "bool") {
            m_converter = __to_array<bool>;
            m_length /= sizeof(bool);
        } else if (data_type == "int8_t") {
            m_converter = __to_array<int8_t>;
            m_length /= sizeof(int8_t);
        } else if (data_type == "int16_t") {
            m_converter = __to_array<int16_t>;
            m_length /= sizeof(int16_t);
        } else if (data_type == "int32_t") {
            m_converter = __to_array<int32_t>;
            m_length /= sizeof(int32_t);
        } else if (data_type == "int64_t") {
            m_converter = __to_array<int64_t>;
            m_length /= sizeof(int64_t);
        } else if (data_type == "uint8_t") {
            m_converter = __to_array<uint8_t>;
            m_length /= sizeof(uint8_t);
        } else if (data_type == "uint16_t") {
            m_converter = __to_array<uint16_t>;
            m_length /= sizeof(uint16_t);
        } else if (data_type == "uint32_t") {
            m_converter = __to_array<uint32_t>;
            m_length /= sizeof(uint32_t);
        } else if (data_type == "uint64_t") {
            m_converter = __to_array<uint64_t>;
            m_length /= sizeof(uint64_t);
        } else if (data_type == "float") {
            m_converter = __to_array<float>;
            m_length /= sizeof(float);
        } else if (data_type == "double") {
            m_converter = __to_array<double>;
            m_length /= sizeof(double);
        } else {
            throw NotImplementedError(data_type);
        }
    }
    Buffer(Buffer &&ohs)
        : m_size(ohs.m_size),
          m_length(ohs.m_length),
          m_data(ohs.m_data),
          m_converter(ohs.m_converter) {
        ohs.m_data = nullptr;
    }

    ~Buffer() {
        delete[] m_data;
        m_data = nullptr;
    }

    void *set(const std::vector<size_t> &shape) {
        m_array = m_converter(shape, m_data);
        return m_data;
    }

    py::array &get() { return m_array; }

    void reset() { m_array = m_converter({m_length}, m_data); }

private:
    size_t    m_size;
    size_t    m_length;
    Bytes    *m_data;
    py::array m_array;
    py::array (*m_converter)(const std::vector<size_t> &, void *);
};

class PythonInterpreter {
public:
    explicit PythonInterpreter(bool init_signal_handlers = true, int argc = 0,
                               const char *const *argv      = nullptr,
                               bool add_program_dir_to_path = true) {
        initialize(init_signal_handlers, argc, argv, add_program_dir_to_path);
    }

    PythonInterpreter(const PythonInterpreter &) = delete;
    PythonInterpreter(PythonInterpreter &&other) noexcept {
        other.is_valid = false;
    }
    PythonInterpreter &operator=(const PythonInterpreter &) = delete;
    PythonInterpreter &operator=(PythonInterpreter &&)      = delete;

    ~PythonInterpreter() { finalize(); }

    void *set_buffer(const std::string         &name,
                     const std::vector<size_t> &shape) {
        auto &buf = m_buffers[name];
        void *p   = buf.set(shape);
        m_py_input.attr("set_buffer_value")(name, buf.get());
        return p;
    }

    std::pair<void *, std::vector<size_t>> get_buffer(const std::string &name) {
        auto &array  = m_buffers[name].get();
        auto  pShape = m_output_buffer_sizes.find(name);
        if (pShape == m_output_buffer_sizes.end()) {
            return std::make_pair(
                array.request().ptr,
                std::vector<size_t>(array.shape(),
                                    array.shape() + array.ndim()));
        } else {
            return std::make_pair(array.request().ptr, pShape->second);
        }
    }

    void run() {
        p_pkg->attr("model")(m_py_input, m_py_output);

        for (auto &kv : m_output_buffer_sizes) {
            kv.second.clear();
            py::tuple shape = m_py_output.attr("get_buffer_shape")(kv.first);

            for (auto &d : shape) {
                kv.second.push_back(d.cast<size_t>());
            }
        }
    }

    void show_buffer(const std::string &name) {
        auto &buf = m_buffers[name];
        p_pkg->attr("show")(buf.get());
    }

private:
    void initialize(bool init_signal_handlers, int argc,
                    const char *const *argv, bool add_program_dir_to_path) {
        py::initialize_interpreter(true, 0, nullptr, true);

        p_pkg = std::make_unique<py::module_>(py::module_::import("model"));
        py::print(p_pkg->attr("__file__"));

        py::tuple spec, output_fields;
        std::tie(m_py_input, m_py_output, spec, output_fields) =
            p_pkg->attr("init")()
                .cast<
                    std::tuple<py::object, py::object, py::tuple, py::tuple>>();

        for (auto d = spec.begin(); d != spec.end(); d++) {
            auto meta = d->cast<py::tuple>();
            m_buffers.insert(std::make_pair(
                meta[0].cast<std::string>(),
                Buffer{meta[2].cast<int>(), meta[1].cast<std::string>()}));
        }

        for (auto d = output_fields.begin(); d != output_fields.end(); d++) {
            const auto name             = d->cast<std::string>();
            m_output_buffer_sizes[name] = {};
            auto &buf                   = m_buffers[name];
            buf.reset();
            m_py_output.attr("set_buffer_value")(name, buf.get());
        }
    }

    void finalize() {
        p_pkg       = nullptr;
        m_py_input  = py::none();
        m_py_output = py::none();

        if (is_valid) {
            py::finalize_interpreter();
            is_valid = false;
        }
    }

    bool                         is_valid = true;
    std::unique_ptr<py::module_> p_pkg;

    std::map<std::string, Buffer>              m_buffers;
    std::map<std::string, std::vector<size_t>> m_output_buffer_sizes;

    py::object m_py_input;
    py::object m_py_output;
};

};  // namespace PyCXpress

#endif  // __PYCXPRESS_HPP__