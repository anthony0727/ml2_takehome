cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(lenet5_libtorch)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

add_executable(lenet5_libtorch ./src/lenet5_libtorch.cpp ./src/lenet5_libtorch.hpp)
target_compile_features(lenet5_libtorch PUBLIC cxx_range_for)
target_link_libraries(lenet5_libtorch ${TORCH_LIBRARIES})

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET lenet5_libtorch
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:lenet5_libtorch>)
endif (MSVC)
