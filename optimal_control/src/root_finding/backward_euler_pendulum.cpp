#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <chrono>
using namespace std::chrono;

// Backward Euler Dynamics Simulation: f_d (x_{k+1}, x_k, u_k) = x_k + h * f(x_{k+1}) - x_{k+1} = 0
// Solve this root finding problem using:
//  - Fixed Point Iteration
//  - Newton's Method

#include <fstream>

void save_csv(const std::string& filename, const Eigen::MatrixXd& mat)
{
    std::ofstream file(filename);
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            file << mat(i, j);
            if (j + 1 < mat.cols())
                file << ",";
        }
        file << "\n";
    }
    file.close();
}

Eigen::Vector2d pendulum_dynamics(const Eigen::Vector2d& x) {
    constexpr double l = 1.0;
    constexpr double g = 9.81;

    double theta     = x(0);
    double theta_dot = x(1);

    double theta_ddot = -(g / l) * std::sin(theta);

    return Eigen::Vector2d(theta_dot, theta_ddot);
}

double backward_euler_step_fixed_point(const Eigen::Vector2d& x0, const double h, Eigen::Vector2d& xn){
    xn = x0;
    double e = (x0 + h * pendulum_dynamics(xn) - xn).norm();
    while (e > 1e-8) {
        xn = x0 + h * pendulum_dynamics(xn);
        e = (x0 + h * pendulum_dynamics(xn) - xn).norm();
    }
    return e;
}

Eigen::Matrix2d pendulum_backward_euler_jacobian(const Eigen::Vector2d& xn, const double h) {
    // d(x_0 + h * f(x_n) - x_n) / d(x_n)
    constexpr double l = 1.0;
    constexpr double g = 9.81;

    double theta = xn(0);

    Eigen::Matrix2d jac(2,2);
    jac << -1, h,
            -h*g*std::cos(theta)/l, -1;

    return jac;

}

double backward_euler_step_newton(const Eigen::Vector2d& x0, const double h, Eigen::Vector2d& xn) {
    xn = x0;
    double e = (x0 + h * pendulum_dynamics(xn) - xn).norm();
    Eigen::Vector2d r = x0 + h * pendulum_dynamics(xn) - xn;
    // std::cout << "r: " << r << "\n";
    while (e > 1e-8) {
        Eigen::Matrix2d J = pendulum_backward_euler_jacobian(xn, h);
        // std::cout << "J: " << J  << "\n";
        Eigen::Vector2d dx = J.ldlt().solve(r);
        // std::cout << "dx: " << dx << "\n";
        xn -= dx;
        // std::cout << "xn: " << xn << "\n";
        r = x0 + h * pendulum_dynamics(xn) - xn;
        // std::cout << "r: " << r << "\n";
        e = r.norm();
        // std::cout << "e: " << e << "\n";

    }
    return e;

}

Eigen::MatrixXd backward_euler_fixed_point(const Eigen::Vector2d& x0, const double Tf, const double h) {
    int num_steps = (int)(Tf / h) + 1;
    Eigen::MatrixXd hist(num_steps, 2);
    hist.row(0) = x0.transpose();

    auto start_t = high_resolution_clock::now();

    for (int i = 1; i < num_steps; i++) {
        Eigen::Vector2d xi = hist.row(i-1).transpose();
        Eigen::Vector2d xin;
        backward_euler_step_fixed_point(xi, h, xin);
        hist.row(i) = xin.transpose();
        std::cout << "\rStep: " << i << "/" << num_steps;
    }

    auto end_t = high_resolution_clock::now();
    std::cout << "\nFPI Total time: " << duration_cast<milliseconds>(end_t - start_t).count() << " ms\n";

    return hist;
}


Eigen::MatrixXd backward_euler_newton(const Eigen::Vector2d& x0, const double Tf, const double h) {
    int num_steps = (int)(Tf / h) + 1;
    Eigen::MatrixXd hist(num_steps, 2);
    hist.row(0) = x0.transpose();

    auto start_t = high_resolution_clock::now();

    for (int i = 1; i < num_steps; i++) {
        Eigen::Vector2d xi = hist.row(i-1).transpose();
        Eigen::Vector2d xin;
        backward_euler_step_newton(xi, h, xin);
        hist.row(i) = xin.transpose();
        std::cout << "\rStep: " << i << "/" << num_steps;
    }

    auto end_t = high_resolution_clock::now();
    std::cout << "\nNewton Total time: " << duration_cast<milliseconds>(end_t - start_t).count() << " ms\n";

    return hist;
}


int main()
{

    double integration_step = 0.01;  // s
    double integration_time = 10;   // s

    Eigen::Vector2d x0;
    x0 << 0.1, 0.0;

    // Test pendulum dynamics
    Eigen::Vector2d dx = pendulum_dynamics(x0);
    std::cout << "dx = " << dx.transpose() << std::endl;

    // Test Fixed point step
    Eigen::Vector2d xn;
    double err=0.0;
    err = backward_euler_step_fixed_point(x0, integration_step, xn);
    std::cout << "xn: " << xn << "\terr: " << err << "\n";

    // Test backward euler forward dynamics using fixed point iteration
    Eigen::MatrixXd hist = backward_euler_fixed_point(x0, integration_time, integration_step);
    save_csv("pendulum_backward_euler_fp.csv", hist);
    
    // Test Newton step
    err = backward_euler_step_newton(x0, integration_step, xn);
    std::cout << "xn: " << xn << "\terr: " << err << "\n";
    
    // Test backward euler forward dynamics using newton
    hist = backward_euler_newton(x0, integration_time, integration_step);
    save_csv("pendulum_backward_euler_newton.csv", hist);
}