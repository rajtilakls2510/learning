cmake_minimum_required(VERSION 3.16)
project(diffusion_rev LANGUAGES CXX CUDA)

set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/media/riju/New Volume/libtorch")

find_package(Torch CONFIG REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})


# Add libtorch flags
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

add_executable(train train.cpp vit.cpp trainer.cpp)
target_link_libraries(train ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET train PROPERTY CXX_STANDARD 17)

add_executable(infer infer.cpp vit.cpp)
target_link_libraries(infer ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET infer PROPERTY CXX_STANDARD 17)
