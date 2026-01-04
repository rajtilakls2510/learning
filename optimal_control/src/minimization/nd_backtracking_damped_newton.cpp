#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
using namespace std;

// double f(const Eigen::Vector2d& x) {
//     double x0 = x(0);
//     double x1 = x(1);
//     return pow(x0, 4) + pow(x1, 4) + pow(x0, 2) + pow(x1, 2) - 2*x0 - 2*x1;
// }

// Eigen::Vector2d grad_f(const Eigen::Vector2d& x) {
//     double x0 = x(0);
//     double x1 = x(1);

//     Eigen::Vector2d g;
//     g(0) = 4.0 * pow(x0, 3) + 2.0 * x0 - 2.0;
//     g(1) = 4.0 * pow(x1, 3) + 2.0 * x1 - 2.0;
//     return g;
// }

// Eigen::Matrix2d hess_f(const Eigen::Vector2d& x) {
//     double x0 = x(0);
//     double x1 = x(1);

//     Eigen::Matrix2d H;
//     H(0,0) = 12.0 * pow(x0, 2) + 2.0;
//     H(0,1) = 0.0;
//     H(1,0) = 0.0;
//     H(1,1) = 12.0 * pow(x1, 2) + 2.0;
//     return H;
// }

double f(const Eigen::Vector2d& x) {
    double x0 = x(0), x1 = x(1);
    return std::pow(x0*x0 + x1 - 11.0, 2)
         + std::pow(x0 + x1*x1 - 7.0, 2);
}

Eigen::Vector2d grad_f(const Eigen::Vector2d& x) {
    double x0 = x(0), x1 = x(1);

    Eigen::Vector2d g;
    g(0) = 4*x0*(x0*x0 + x1 - 11) + 2*(x0 + x1*x1 - 7);
    g(1) = 2*(x0*x0 + x1 - 11) + 4*x1*(x0 + x1*x1 - 7);
    return g;
}

Eigen::Matrix2d hess_f(const Eigen::Vector2d& x) {
    double x0 = x(0), x1 = x(1);

    Eigen::Matrix2d H;
    H(0,0) = 12*x0*x0 + 4*x1 - 42;
    H(1,1) = 12*x1*x1 + 4*x0 - 26;
    H(0,1) = H(1,0) = 4*(x0 + x1);
    return H;
}


bool is_pos_def(const Eigen::Matrix2d& H) {
    Eigen::LLT<Eigen::Matrix2d> llt(H);
    return llt.info() == Eigen::Success;
}

Eigen::Vector2d backtracking_regularized_newton_step(
    const Eigen::Vector2d& x
) {
    Eigen::Vector2d g = grad_f(x);
    Eigen::Matrix2d H = hess_f(x);

    double beta = 1e-6;
    Eigen::Matrix2d I = Eigen::Matrix2d::Identity();

    // Regularize Hessian until PD
    while (!is_pos_def(H)) {
        H += beta * I;
        beta *= 10.0;
    }

    // Newton direction
    Eigen::Vector2d dx = -H.ldlt().solve(g);

    // Backtracking line search (Armijo)
    double alpha = 1.0;
    double b = 1e-4;
    double c = 0.5;

    while (f(x + alpha*dx) > f(x) + b * alpha * g.dot(dx)) {
        alpha *= c;
    }

    return x + alpha * dx;
}

int main() {

    // ---- save contour data ----
    std::ofstream ffile("surface.csv");
    double xmin = -5.0, xmax = 5.0;
    int N = 200;

    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j <= N; ++j) {
            double x = xmin + (xmax - xmin) * i / N;
            double y = xmin + (xmax - xmin) * j / N;
            ffile << x << "," << y << "," << f({x,y}) << "\n";
        }
    }
    ffile.close();

    // ---- Newton iterates ----
    Eigen::Vector2d x(-1.0, -4.5);

    std::ofstream nfile("newton.csv");
    for (int i = 0; i < 50; ++i) {
        nfile << x(0) << "," << x(1) << "," << f(x) << "\n";
        x = backtracking_regularized_newton_step(x);
    }
    nfile.close();

    std::cout << "Data written to surface.csv and newton.csv\n";
}


