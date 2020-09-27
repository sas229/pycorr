cmake_minimum_required(VERSION 3.1)
project(pycorr_extensions)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DEIGEN_DONT_PARALLELIZE")
message(${CMAKE_CXX_FLAGS})

## Find installed libraries.
#find_package(pybind11 REQUIRED)
find_package (Eigen3 3.3 REQUIRED)

## Included libraries.
#add_subdirectory(lib/Eigen)
add_subdirectory(lib/pybind11)

pybind11_add_module(pycorr_extensions pycorr/pycorr_extensions.cpp)
target_link_libraries(pycorr_extensions PUBLIC Eigen3::Eigen)
