#include <doctest/doctest.h>
#include <tensorflow_cpy/framework.h>
#include <tensorflow_cpy/version.h>

#include <string>

TEST_CASE("TensorflowCpy") { CHECK(tensorflow_cpy::main_whole_flow(0, nullptr) == 1); }

TEST_CASE("TensorflowCpy version") {
    static_assert(std::string_view(TENSORFLOWCPY_VERSION) == std::string_view("0.0.1"));
    CHECK(std::string(TENSORFLOWCPY_VERSION) == std::string("0.0.1"));
}
