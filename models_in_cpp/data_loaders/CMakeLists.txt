# Load shared dependencies
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/Dependencies.cmake)

# Create a static library named "data_loaders"
add_library(data_loaders
    mnist_loader.cpp
)

# Make the include path visible to anything that links datasets
target_include_directories(data_loaders
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(data_loaders
    PUBLIC
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
)

# Ensure C++17
set_property(TARGET data_loaders PROPERTY CXX_STANDARD 17)