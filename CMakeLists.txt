cmake_minimum_required(VERSION 3.18)
project(my_deep_lib LANGUAGES CXX CUDA)

# 设置 C++ 和 CUDA 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找依赖
find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

# 使用 pip 安装的 pybind11
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED)

# CUDA 架构设置（根据你的 GPU 调整）
# RTX 40xx: 89, RTX 30xx: 86, RTX 20xx: 75, GTX 10xx: 61
set(CMAKE_CUDA_ARCHITECTURES 75 86)

# 头文件目录
include_directories(
    ${CMAKE_SOURCE_DIR}/cuda_src
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS}
)

# CUDA 源文件
set(CUDA_SOURCES
    cuda_src/tensor.cu
    cuda_src/conv_layer.cu
    cuda_src/layers.cu
    cuda_src/activations.cu
    cuda_src/elementwise.cu
)

# Pybind11 模块
pybind11_add_module(my_deep_lib
    ${CUDA_SOURCES}
    pybind/pybind_wrapper.cpp
)

# 链接库
target_link_libraries(my_deep_lib PRIVATE
    CUDA::cudart
    CUDA::cublas
)

# 设置输出目录
set_target_properties(my_deep_lib PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python
)

# 编译选项
target_compile_options(my_deep_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --expt-relaxed-constexpr
        -Xcompiler -fPIC
    >
)