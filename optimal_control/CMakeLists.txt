cmake_minimum_required(VERSION 3.18)

project(optimal_control LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)

# ========== ROOT FINDING =========

add_executable(backward_euler_pendulum
    src/root_finding/backward_euler_pendulum.cpp
)

target_link_libraries(backward_euler_pendulum
    Eigen3::Eigen
)

# ========= Minimization ===========


add_executable(newton
    src/minimization/newton.cpp
)

add_executable(regularized_newton
    src/minimization/regularized_newton.cpp
)

add_executable(backtracking_damped_newton
    src/minimization/backtracking_damped_newton.cpp
)

add_executable(nd_backtracking_damped_newton
    src/minimization/nd_backtracking_damped_newton.cpp
)
target_link_libraries(nd_backtracking_damped_newton
    Eigen3::Eigen
)