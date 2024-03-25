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

        if (data_type == "bool_") {
            m_converter = __to_array<bool>;
            m_length /= sizeof(bool);
        } else if (data_type == "float_") {
            m_converter = __to_array<float>;
            m_length /= sizeof(float);
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
        m_py_input.attr("set")(name, buf.get());
        return p;
    }

    std::pair<void *, std::vector<size_t>> get_buffer(const std::string &name) {
        auto &array = m_buffers[name].get();
        return std::make_pair(
            array.request().ptr,
            std::vector<size_t>(array.shape(), array.shape() + array.ndim()));
    }

    void run() {
        auto &buf = m_buffers["output_a"];
        buf.reset();
        m_py_output.attr("set_buffer")("output_a", buf.get());

        p_pkg->attr("model")(m_py_input, m_py_output);

        py::tuple py_shape = m_py_output.attr("get_shape")("output_a");
        auto     &shape    = m_output_buffer_sizes["output_a"];
        shape.clear();
        for (auto d = py_shape.begin(); d != py_shape.end(); d++) {
            shape.push_back(d->cast<int>());
        }
        set_buffer("output_a", shape);
    }

    void show_buffer(const std::string &name) {
        auto &buf = m_buffers[name];
        p_pkg->attr("show")(buf.get());
    }

private:
    void initialize(bool init_signal_handlers, int argc,
                    const char *const *argv, bool add_program_dir_to_path) {
        py::initialize_interpreter(true, 0, nullptr, true);

        m_buffers.insert(std::make_pair("input_a", Buffer{1000, "float_"}));
        m_buffers.insert(std::make_pair("input_b", Buffer{1000, "float_"}));
        m_buffers.insert(std::make_pair("output_a", Buffer{1000, "float_"}));


        p_pkg = std::make_unique<py::module_>(py::module_::import("model"));

        py::print(p_pkg->attr("__file__"));

        m_py_input  = p_pkg->attr("InputDataSet")();
        m_py_output = p_pkg->attr("OutputDataSet")();

        m_py_output.attr("set_buffer")("output_a", m_buffers["output_a"].get());
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