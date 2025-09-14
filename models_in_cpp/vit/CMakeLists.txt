cmake_minimum_required(VERSION 3.16)
project(vit LANGUAGES CXX CUDA)

# Load shared dependencies
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/Dependencies.cmake)

# Add data_loaders as a subdirectory (relative to vit/)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../data_loaders
                 ${CMAKE_CURRENT_BINARY_DIR}/data_loaders_build)

add_executable(train train.cpp vit.cpp trainer.cpp)
target_link_libraries(train ${TORCH_LIBRARIES} ${OpenCV_LIBS} data_loaders)
set_property(TARGET train PROPERTY CXX_STANDARD 17)

add_executable(infer infer.cpp vit.cpp)
target_link_libraries(infer ${TORCH_LIBRARIES} ${OpenCV_LIBS} data_loaders)
set_property(TARGET infer PROPERTY CXX_STANDARD 17)
