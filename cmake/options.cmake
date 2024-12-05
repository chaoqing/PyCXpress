# Detecting OS
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
  set(IS_WINDOWS ON)
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  set(IS_OSX ON)
endif()

if(${CMAKE_SYSTEM} STREQUAL "Linux-4.4.38-tegra")
  set(IS_JETSON ON)
endif()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
  set(IS_LINUX ON)
endif()

# Detecting Compiler
set(COMPILER_IS_MSVC OFF)
set(COMPILER_IS_GCC OFF)
set(COMPILER_IS_CLANG OFF)

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC")
  set(COMPILER_IS_MSVC ON)
endif()

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  set(COMPILER_IS_GCC ON)
endif()

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(COMPILER_IS_CLANG ON)
endif()

# Configuring Compiler options
set(DEFAULT_COMPILE_DEFINITIONS)
set(DEFAULT_COMPILE_OPTIONS)
set(DEFAULT_LINKER_OPTIONS)

# MSVC compiler options
if(COMPILER_IS_MSVC)
  list(
    APPEND DEFAULT_COMPILE_DEFINITIONS
    _SCL_SECURE_NO_WARNINGS # Suspend potentially unsafe methods warning in the Standard C++ Library
    _CRT_SECURE_NO_WARNINGS # Suspend potentially unsafe methods warning in the CRT Library
  )

  list(APPEND DEFAULT_COMPILE_OPTIONS /MP # -> build with multiple processes
       /W3 # -> warning level 3
  )

  list(APPEND DEFAULT_LINKER_OPTIONS shlwapi.lib)
endif()

# GCC and Clang compiler options
if(COMPILER_IS_GCC OR COMPILER_IS_CLANG)
  list(APPEND DEFAULT_COMPILE_OPTIONS -Wall)
endif()

# Use pthreads on mingw and linux
if(COMPILER_IS_GCC OR IS_LINUX)
  list(APPEND DEFAULT_LINKER_OPTIONS -pthread)
endif()
