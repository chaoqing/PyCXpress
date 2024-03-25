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

    python.show_buffer("input_a");
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