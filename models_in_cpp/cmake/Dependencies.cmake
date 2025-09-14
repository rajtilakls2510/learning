
# Tell CMake where to find libtorch
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/media/riju/New Volume2/libtorch")

# Torch
find_package(Torch CONFIG REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)

# Global include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Set flags (Torch requires some)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
