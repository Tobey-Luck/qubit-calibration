cmake_minimum_required(VERSION 3.15)
project(qubit_calibration_cpp CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(PYBIND11_FINDPYTHON ON)
find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
pybind11_add_module(cpp_solver
    src/bloch_solver.cpp
    src/bindings.cpp
)
target_include_directories(cpp_solver PRIVATE include)
add_executable(test_bloch_cpp
    src/bloch_solver.cpp
    src/test_bloch_main.cpp
)
target_include_directories(test_bloch_cpp PRIVATE include)