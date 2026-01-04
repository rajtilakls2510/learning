cmake_minimum_required(VERSION 3.18)
project(ransac LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ================= CPU Version ================

# Create the shared library
add_library(ransac_cpu SHARED
    src/cpu/adam.cpp
    src/cpu/model.cpp
)

# Public include directory for consumers
target_include_directories(ransac_cpu
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Example program
add_executable(adam_cpu
    example/adam_test.cpp
)


# Link the example to the Ransac shared library
target_link_libraries(adam_cpu
    PRIVATE ransac_cpu
)

add_executable(model_cpu
    example/model_test.cpp
)

target_link_libraries(model_cpu
    PRIVATE ransac_cpu
)

# ================ CUDA version =====================
# Specify CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)

add_library(ransac_cuda SHARED
    src/cuda/adam.cu
    src/cuda/model.cu
)
set_target_properties(ransac_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ransac_cuda PRIVATE cuda cudart)
target_include_directories(ransac_cuda
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/include
)

# Example program
add_executable(adam_cuda
    example/adam_test.cpp
)

# Link the example to the Ransac shared library
target_link_libraries(adam_cuda
    PRIVATE ransac_cuda
)

add_executable(model_cuda
    example/model_test.cpp
)

target_link_libraries(model_cuda
    PRIVATE ransac_cuda
)





