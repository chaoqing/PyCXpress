#include "PyCXpress.hpp"

namespace pcx = PyCXpress;

void show_test(pcx::PythonInterpreter &guard)
{
    std::vector<float> data(6);
    for (size_t i = 0; i < 6; i++)
    {
        data[i] = i;
    }

    memcpy(guard.set_buffer("input_a", {3, 2}), data.data(), data.size() * sizeof(float));

    guard.show_buffer("input_a");
}

int main(int argc, char *argv[])
{
    pcx::PythonInterpreter guard{};
    int loop_times = 3;

    while (loop_times--)
    {
        std::cout << "looping " << loop_times << std::endl;
        show_test(guard);
    }

    return 0;
}