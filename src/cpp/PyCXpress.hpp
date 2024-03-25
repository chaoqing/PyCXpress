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
        std::cout << stride[0] << " " << stride[1] << std::endl;
        return py::array_t<T>{shape, std::move(stride), (T *)(data),
                              py::none()};
    }

public:
    Buffer() : m_size(0), m_data(nullptr), m_converter(nullptr) {}
    Buffer(size_t size, const std::string &data_type) : m_size(size) {
        m_data = new Bytes[m_size];

        if (data_type == "bool_") {
            m_converter = __to_array<bool>;
        } else if (data_type == "float_") {
            m_converter = __to_array<float>;
        } else {
            throw NotImplementedError(data_type);
        }
    }
    Buffer(Buffer &&ohs)
        : m_size(ohs.m_size), m_data(ohs.m_data), m_converter(ohs.m_converter) {
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

private:
    size_t    m_size;
    Bytes    *m_data;
    py::array m_array;
    py::array (*m_converter)(const std::vector<size_t> &, void *);
};

class PythonInterpreter {
public:
    explicit PythonInterpreter(bool init_signal_handlers = true, int argc = 0,
                               const char *const *argv      = nullptr,
                               bool add_program_dir_to_path = true) {
        py::initialize_interpreter(true, 0, nullptr, true);

        m_buffers.insert(std::make_pair("input_a", Buffer{1000, "float_"}));

        p_pkg = std::make_unique<py::module_>(py::module_::import("model"));

        py::print(p_pkg->attr("__file__"));

        c_show_func = p_pkg->attr("show");
    }

    PythonInterpreter(const PythonInterpreter &) = delete;
    PythonInterpreter(PythonInterpreter &&other) noexcept {
        other.is_valid = false;
    }
    PythonInterpreter &operator=(const PythonInterpreter &) = delete;
    PythonInterpreter &operator=(PythonInterpreter &&)      = delete;

    ~PythonInterpreter() {
        p_pkg = nullptr;

        if (is_valid) {
            py::finalize_interpreter();
            is_valid = false;
        }
    }

    void *set_buffer(const std::string         &name,
                     const std::vector<size_t> &shape) {
        auto &buf = m_buffers[name];
        return buf.set(shape);
    }

    void show_buffer(const std::string &name) {
        auto &buf = m_buffers[name];
        c_show_func(buf.get());
    }

private:
    bool                         is_valid = true;
    py::object                   c_show_func;
    std::unique_ptr<py::module_> p_pkg;

    std::map<std::string, Buffer> m_buffers;
};

};  // namespace PyCXpress

#endif  // __PYCXPRESS_HPP__