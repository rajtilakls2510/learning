cmake_minimum_required(VERSION 3.16)
project(diff LANGUAGES CXX CUDA)

# Load shared dependencies
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/Dependencies.cmake)

# Add data_loaders as a subdirectory (relative to vit/)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../data_loaders
                 ${CMAKE_CURRENT_BINARY_DIR}/data_loaders_build)


add_executable(net_test net_test.cpp network.cpp)
target_link_libraries(net_test ${TORCH_LIBRARIES})
set_property(TARGET net_test PROPERTY CXX_STANDARD 17)

add_executable(train train.cpp network.cpp trainer.cpp)
target_link_libraries(train ${TORCH_LIBRARIES} ${OpenCV_LIBS} data_loaders)
set_property(TARGET train PROPERTY CXX_STANDARD 17)

# add_executable(sample sampler.cpp network.cpp)
# target_link_libraries(sample ${TORCH_LIBRARIES} ${OpenCV_LIBS} data_loaders)
# set_property(TARGET sample PROPERTY CXX_STANDARD 17)
