cmake_minimum_required(VERSION 3.18)

project(optimal_control LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

add_executable(backward_euler_pendulum
    src/root_finding/backward_euler_pendulum.cpp
)

target_link_libraries(backward_euler_pendulum
    Eigen3::Eigen
)