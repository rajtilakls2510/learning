cmake_minimum_required(VERSION 3.18)
project(learning_cuda LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 86)

add_executable(sum src/sum.cu)
target_link_libraries(sum cuda cudart)

add_executable(reduce_sum src/reduce_sum.cu)
target_link_libraries(reduce_sum cuda cudart)

add_executable(transpose src/transpose.cu)
target_link_libraries(transpose cuda cudart)

add_executable(matrix_multiply src/matrix_multiply.cu)
target_link_libraries(matrix_multiply cuda cudart)