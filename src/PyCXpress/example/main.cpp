#include <algorithm>
#include <iostream>
#include <iterator>
#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <PyCXpress/core.hpp>
#include <PyCXpress/utils.hpp>

namespace pcx = PyCXpress;

void show_test(pcx::Model &model) {
    std::vector<double> data(12);
    for (size_t i = 0; i < 12; i++) {
        data[i] = i;
    }

    std::vector<uint8_t> shape   = {3, 4};
    void                *pBuffer = nullptr;
    size_t               nBytes  = 0;
    std::tie(pBuffer, nBytes) = model.set_buffer("data_to_be_reshaped", {12});
    assert(data.size() * sizeof(double) == nBytes);

    memcpy(pBuffer, data.data(), nBytes);
    memcpy(model.set_buffer("new_2d_shape", {2}).first, shape.data(),
           shape.size() * sizeof(uint8_t));

    model.run();

    // test retrieve output tensor
    void               *p = nullptr;
    std::vector<size_t> new_shape;
    pcx::BufferPtr      buf;
    std::tie(buf, new_shape) = model.get_buffer("output_a");
    std::tie(p, nBytes)      = buf;

    std::cout << "output shape: ";
    std::copy(new_shape.begin(), new_shape.end(),
              std::ostream_iterator<size_t>(std::cout, ", "));
    std::cout << std::endl;

    size_t size = std::accumulate(new_shape.begin(), new_shape.end(), 1,
                                  std::multiplies<int>());
    assert(nBytes == sizeof(double) * size);
    std::cout << "output data: ";
    std::copy((double *)p, (double *)p + size,
              std::ostream_iterator<double>(std::cout, ", "));
    std::cout << std::endl;

    // test retrieve input tensor
    std::tie(buf, new_shape) = model.get_buffer("new_2d_shape");
    assert(new_shape.size() == 1 && new_shape.front() == shape.size());
    assert(buf.second == shape.size() * sizeof(uint8_t));
    assert(0 == std::memcmp(buf.first, shape.data(), buf.second));
}

int main(int argc, char *argv[]) {
    auto &python     = utils::Singleton<pcx::PythonInterpreter>::Instance();
    auto &model0     = python.create_model("model.Model");
    auto &model1     = python.create_model("model.Model", "odd");
    int   loop_times = 3;


    while (loop_times--) {
        std::cout << "looping " << loop_times << std::endl;
        if (loop_times % 2 == 0) {
            show_test(model0);
        } else {
            show_test(model1);
        }
    }

    return 0;
}
