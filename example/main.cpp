#include <algorithm>
#include <iostream>
#include <iterator>

#include "PyCXpress.hpp"
#include "Utils.hpp"

namespace pcx = PyCXpress;

void show_test(pcx::PythonInterpreter &python) {
    std::vector<float> data(6);
    for (size_t i = 0; i < 6; i++) {
        data[i] = i;
    }

    memcpy(python.set_buffer("input_a", {3, 2}), data.data(),
           data.size() * sizeof(float));
    memcpy(python.set_buffer("input_b", {3, 2}), data.data(),
           data.size() * sizeof(float));

    python.run();

    void               *p = nullptr;
    std::vector<size_t> shape;
    std::tie(p, shape) = python.get_buffer("output_a");

    std::cout << "output shape: ";
    std::copy(shape.begin(), shape.end(),
              std::ostream_iterator<double>(std::cout, ", "));
    std::cout << std::endl;

    size_t size =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    std::cout << "output data: ";
    std::copy((float *)p, (float *)p + size,
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