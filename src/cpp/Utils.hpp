#ifndef __PYCXPRESS_UTILS__
#define __PYCXPRESS_UTILS__

#include <stdexcept>
#include <string>

namespace utils {
class NotImplementedError : public std::logic_error {
public:
    NotImplementedError(const std::string &what_arg)
        : std::logic_error(what_arg) {}
    NotImplementedError(const char *what_arg) : std::logic_error(what_arg) {}
};

template <typename T>
class Singleton {
public:
    static T &Instance() {
        static T instance;  // Guaranteed to be destroyed.
                            // Instantiated on first use.
        return instance;
    }

    Singleton(Singleton const &)      = delete;  // Prevent copying
    void operator=(Singleton const &) = delete;  // Prevent assignment

protected:
    Singleton() {}  // Protected constructor
    ~Singleton() {}
};
};  // namespace utils

#endif  // __PYCXPRESS_UTILS__