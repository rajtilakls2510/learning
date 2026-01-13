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

add_executable(equality_constraints
    src/minimization/equality_constraints.cpp
)
target_link_libraries(equality_constraints
    Eigen3::Eigen
)

add_executable(inequality_constraints
    src/minimization/inequality_constraints.cpp
)
target_link_libraries(inequality_constraints
    Eigen3::Eigen
)

add_executable(merit_functions
    src/minimization/merit_functions.cpp
)
target_link_libraries(merit_functions
    Eigen3::Eigen
)

add_executable(al_gn_solver
    src/minimization/al_gn_solver.cpp
)
target_link_libraries(al_gn_solver
    Eigen3::Eigen
)

add_executable(al_gn_solver_complex
    src/minimization/al_gn_solver_complex.cpp
)
target_link_libraries(al_gn_solver_complex
    Eigen3::Eigen
)



# Additional resources like images and fonts
configure_file(res/Roboto-Regular.ttf Roboto-Regular.ttf COPYONLY)

# Add executable to open game environment
find_package(SFML COMPONENTS graphics window system REQUIRED)

add_executable(simulate_ip 
    src/mpc/simulate_ip.cpp
    # src/mpc/pid.cpp
    # src/mpc/inverted_pendulum.cpp
    # src/mpc/tools.cpp
)
target_link_libraries(simulate_ip
    Eigen3::Eigen sfml-graphics sfml-window sfml-system
)

add_executable(ip_solver 
    src/mpc/ip_solver.cpp
)
target_link_libraries(ip_solver
    Eigen3::Eigen 
)