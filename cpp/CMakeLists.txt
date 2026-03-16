cmake_minimum_required(VERSION 3.15)
project(qubit_calibration_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release build by default for performance
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")

# ---- Find pybind11 ----
# pybind11 can be found via pip-installed package or system install
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # Fallback: find via Python
    execute_process(
        COMMAND python3 -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})
endif()

# ---- Build pybind11 extension module ----
pybind11_add_module(cpp_solver
    src/bloch_solver.cpp
    src/bindings.cpp
)

target_include_directories(cpp_solver PRIVATE include)

# ---- Optional: standalone C++ test binary ----
add_executable(test_bloch_cpp
    src/bloch_solver.cpp
    src/test_bloch_main.cpp
)
target_include_directories(test_bloch_cpp PRIVATE include)
