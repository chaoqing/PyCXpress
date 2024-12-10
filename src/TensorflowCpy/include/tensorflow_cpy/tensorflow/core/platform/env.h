
#ifndef TENSORFLOW_CPY_CORE_PLATFORM_ENV_H_
#define TENSORFLOW_CPY_CORE_PLATFORM_ENV_H_

#include <stdint.h>

#include <string>
#include <vector>

#include "../../core/platform/status.h"


// clang-format off
namespace tensorflow_cpy {
namespace tensorflow {
  using namespace ::tensorflow;

/// \brief An interface used by the tensorflow implementation to
/// access operating system functionality like the filesystem etc.
///
/// Callers may wish to provide a custom Env object to get fine grain
/// control.
///
/// All Env implementations are safe for concurrent access from
/// multiple threads without any external synchronization.
class Env {
 public:
  Env();
  virtual ~Env() = default;

  /// \brief Returns a default environment suitable for the current operating
  /// system.
  ///
  /// Sophisticated users may wish to provide their own Env
  /// implementation instead of relying on this default environment.
  ///
  /// The result of Default() belongs to this library and must never be deleted.
  static Env* Default();

  /// \brief Returns the file system schemes registered for this Env.
  virtual Status GetRegisteredFileSystemSchemes(
      std::vector<std::string>* schemes);


  Status SetOption(const std::string& scheme, const std::string& key,
                   const std::string& value);

  Status SetOption(const std::string& scheme, const std::string& key,
                   const std::vector<std::string>& values);

  Status SetOption(const std::string& scheme, const std::string& key,
                   const std::vector<int64_t>& values);

  Status SetOption(const std::string& scheme, const std::string& key,
                   const std::vector<double>& values);

  /// \brief Return the runfiles directory if running under bazel. Returns
  /// the directory the executable is located in if not running under bazel.
  virtual std::string GetRunfilesDir() = 0;

  /// Sleeps/delays the thread for the prescribed number of micro-seconds.
  virtual void SleepForMicroseconds(int64_t micros) = 0;

  /// Returns the process ID of the calling process.
  int32_t GetProcessId();

  // Returns the thread id of calling thread.
  // Posix: Returns pthread id which is only guaranteed to be unique within a
  //        process.
  // Windows: Returns thread id which is unique.
  virtual int32_t GetCurrentThreadId() = 0;

  // Copies current thread name to "name". Returns true if success.
  virtual bool GetCurrentThreadName(std::string* name) = 0;

  // \brief Load a dynamic library.
  //
  // Pass "library_filename" to a platform-specific mechanism for dynamically
  // loading a library.  The rules for determining the exact location of the
  // library are platform-specific and are not documented here.
  //
  // On success, returns a handle to the library in "*handle" and returns
  // OK from the function.
  // Otherwise returns nullptr in "*handle" and an error status from the
  // function.
  virtual Status LoadDynamicLibrary(const char* library_filename,
                                    void** handle) = 0;

  // \brief Get a pointer to a symbol from a dynamic library.
  //
  // "handle" should be a pointer returned from a previous call to LoadLibrary.
  // On success, store a pointer to the located symbol in "*symbol" and return
  // OK from the function. Otherwise, returns nullptr in "*symbol" and an error
  // status from the function.
  virtual Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                                      void** symbol) = 0;

  // \brief build the name of dynamic library.
  //
  // "name" should be name of the library.
  // "version" should be the version of the library or NULL
  // returns the name that LoadLibrary() can use
  virtual std::string FormatLibraryFileName(const std::string& name,
                                            const std::string& version) = 0;

  // Returns a possible list of local temporary directories.
  virtual void GetLocalTempDirectories(std::vector<std::string>* list) = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(Env);
};


/// \brief Cross-platform setenv.
///
/// Since setenv() is not available on windows, we provide an
/// alternative with platform specific implementations here.
int setenv(const char* name, const char* value, int overwrite);

/// Cross-platform unsetenv.
int unsetenv(const char* name);

}  // namespace tensorflow
}  // namespace tensorflow_cpy
// clang-format on


#endif  // TENSORFLOW_CPY_CORE_PLATFORM_ENV_H_
