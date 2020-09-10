cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(lenet5_libtorch)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(main src/lenet5_libtorch.cpp src/lenet5_libtorch.hpp)
target_link_libraries(main "${TORCH_LIBRARIES}")
set_property(TARGET main PROPERTY CXX_STANDARD 14)
