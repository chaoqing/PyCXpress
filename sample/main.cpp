#include <fmt/format.h>
#include <tensorflow_cpy/framework.h>
#include <tensorflow_cpy/version.h>

#include <cxxopts.hpp>
#include <iostream>
#include <magic_enum.hpp>
#include <string>

/**  Language codes to be used with the TensorflowCpy class */
enum class ExampleCase {
    WHOLE_FLOW,
};

/**
 * @brief A class for saying hello in multiple languages
 */
class TensorflowCpy {
    int (*func)(int argc, char** argv) = nullptr;

public:
    /**
     * @brief Creates a new tensorflow_cpy
     * @param name the name to greet
     */
    TensorflowCpy(std::string name);

    /**
     * @brief Creates a localized string containing the greeting
     * @param argc the language to greet in
     * @param argv the language to greet in
     * @return a string containing the greeting
     */
    int main(int argc, char** argv) const;
};


TensorflowCpy::TensorflowCpy(std::string name) {
    const auto c = magic_enum::enum_cast<ExampleCase>(name, magic_enum::case_insensitive);
    if (c.has_value()) {
        std::cout << fmt::format("Using function main_{}!", name) << std::endl;
    } else {
        throw std::runtime_error(fmt::format("{} not found!", name));
    }
    switch (c.value()) {
        case ExampleCase::WHOLE_FLOW: {
            func = tensorflow_cpy::main_whole_flow;
            break;
        }
        default: {
        }
    }
}

int TensorflowCpy::main(int argc, char** argv) const {
    return (argc == 0 || func == nullptr) ? 1 : (*func)(argc, argv);
}


auto main(int argc, char** argv) -> int {
    cxxopts::Options options(*argv, "A program to welcome the world!");
    options.allow_unrecognised_options();

    std::string name;

    // clang-format off
  options.add_options()
    ("h,help", "Show help")
    ("v,version", "Print the current version number")
    ("n,name", "Name to greet", cxxopts::value(name)->default_value("whole_flow"))
  ;
    // clang-format on

    auto result = options.parse(argc, argv);

    if (result["help"].as<bool>()) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (result["version"].as<bool>()) {
        std::cout << "TensorflowCpy, version " << TENSORFLOWCPY_VERSION << std::endl;
        return 0;
    }

    std::vector<char*> main_argv;
    main_argv.push_back(const_cast<char*>(name.data()));
    for (auto& i : result.unmatched()) {
        main_argv.push_back(const_cast<char*>(i.data()));
    }
    return TensorflowCpy{name}.main(main_argv.size(), main_argv.data());
}
