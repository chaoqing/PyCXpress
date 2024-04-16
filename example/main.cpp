#include <algorithm>
#include <iostream>
#include <iterator>
#define PYBIND11_DETAILED_ERROR_MESSAGES

#include "PyCXpress.hpp"
#include "Utils.hpp"

namespace pcx = PyCXpress;

void show_test(pcx::PythonInterpreter &python) {
    std::vector<double> data(12);
    for (size_t i = 0; i < 12; i++) {
        data[i] = i;
    }

    std::vector<uint8_t> shape = {3, 4};
    memcpy(python.set_buffer("data_to_be_reshaped", {12}), data.data(),
           data.size() * sizeof(double));
    memcpy(python.set_buffer("new_2d_shape", {2}), shape.data(),
           shape.size() * sizeof(uint8_t));

    python.run();

    void               *p = nullptr;
    std::vector<size_t> new_shape;
    std::tie(p, new_shape) = python.get_buffer("output_a");

    std::cout << "output shape: ";
    std::copy(new_shape.begin(), new_shape.end(),
              std::ostream_iterator<size_t>(std::cout, ", "));
    std::cout << std::endl;

    size_t size = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                  std::multiplies<int>());
    std::cout << "output data: ";
    std::copy((double *)p, (double *)p + size,
              std::ostream_iterator<double>(std::cout, ", "));
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    auto &python     = utils::Singleton<pcx::PythonInterpreter>::Instance();
    int   loop_times = 3;

    while (loop_times--) {
        std::cout << "looping " << loop_times << std::endl;
        show_test(python);
    }

    return 0;
}