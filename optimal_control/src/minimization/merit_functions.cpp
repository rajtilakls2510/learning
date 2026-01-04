#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
using namespace std;
using namespace std::chrono;

// Solve min_x 0.5 * (x - [1; 0])' @ Q @ (x - [1; 0]) s.t. x[0]^2 + 2 * x[0] - x[1] using:
//  - Gauss-Newton + Line search using merit functions

double f(const Eigen::Vector2d& x, const Eigen::Matrix2d& Q) {
    Eigen::Vector2d x_t;
    x_t << 1.0 , 0.0;
    return 0.5 * (x - x_t).transpose() * Q * (x - x_t);
}

Eigen::Vector2d df(const Eigen::Vector2d& x, const Eigen::Matrix2d& Q) {
    Eigen::Vector2d x_t;
    x_t << 1.0 , 0.0;
    return Q * (x - x_t);   
}

Eigen::Matrix2d ddf(const Eigen::Vector2d& x, const Eigen::Matrix2d& Q) {
    return Q;
}

double c(const Eigen::Vector2d& x) {
    return pow(x(0), 2) + 2 * x(0) - x(1);
}

Eigen::Vector2d dc(const Eigen::Vector2d& x) {
    Eigen::Vector2d d;
    d << 2*x(0) + 2, -1;
    return d;
}

// ==================== Commonly used merit ============

// double p(const Eigen::Vector2d& x, const Eigen::Matrix2d& Q, const double lamda, const double rho) {
//     return f(x, Q) + rho * std::abs(c(x)); // TODO: For nd, take c(x).norm();
// }

// Eigen::Vector3d dp(const Eigen::Vector2d& x, const Eigen::Matrix2d& Q, const double lamda, double rho) {

//     double cx = c(x);

//     double s = 0.0;
//     if (cx > 0.0)      
//         s =  1.0;
//     else if (cx < 0.0) 
//         s = -1.0;
//     else               
//         s =  0.0;   // subgradient at 0

//     // return df(x, Q) + rho * s * dc(x);

//     // Gradient w.r.t x
//     Eigen::Vector2d grad_x = df(x, Q) + rho * s * dc(x);

//     // Stack [grad_x ; 0]
//     Eigen::Vector3d grad;
//     grad << grad_x, 0.0;

//     return grad;

//     // TODO: For nd,
//     // Eigen::VectorXd cx = c(x);           // m×1
//     // Eigen::MatrixXd C = dc(x);           // m×n
//     // return df(x) + rho * C.transpose() * cx.array().sign().matrix();
// }

// ============= Augmented Lagrangian Merit ==================

double p(const Eigen::Vector2d& x, const Eigen::Matrix2d& Q, const double lamda, const double rho) {
    return f(x, Q) + lamda * c(x) + 0.5 * rho * (c(x) * c(x));  // TODO: For nd, Second term: lambda^T @ c(x), Third term dot(c(x), c(x))
}

Eigen::Vector3d dp(const Eigen::Vector2d& x, const Eigen::Matrix2d& Q, const double lamda, double rho) {
    Eigen::Vector2d m = df(x,Q) + dc(x) * (lamda + rho * c(x));  // TODO: For nd, f(x) + dc^T @ (l + rho * c(x))

    Eigen::Vector3d grad;
    grad << m, c(x);
    return grad;
}


Eigen::Vector3d gauss_newton_step(const Eigen::Vector3d& xl0, const Eigen::Matrix2d& Q) {
    // Extract Vector2d x = xl0[:2]
    Eigen::Vector2d x = xl0.head<2>();
    double lamda = xl0(2);
    Eigen::Matrix2d H = ddf(x, Q);// + ddcl(x, lamda);
    Eigen::Vector2d C = dc(x);

    Eigen::Matrix3d KKT;    // KKT = [H C'; C 0]
    KKT.setZero();
    // Top-left: H
    KKT.block<2,2>(0,0) = H;
    // Top-right: Cᵀ
    KKT.block<2,1>(0,2) = C.transpose();
    // Bottom-left: C
    KKT.block<1,2>(2,0) = C;

    Eigen::Vector3d b;
    Eigen::Vector2d dL = -df(x, Q) - C * lamda;
    // Place b[:2] = dL 
    b.head<2>() = dL;
    b(2) = -c(x);

    Eigen::Vector3d dz = KKT.ldlt().solve(b);
    return dz;  // Returns change in variables
}


int main() {

    // ---- Problem setup ----
    Eigen::Matrix2d Q;
    Q << 0.5, 0.0,
         0.0, 1.0;

    // ---- 1. Sample cost function on a grid ----
    double xmin = -2.0, xmax = 2.0;
    double ymin = -2.0, ymax = 2.0;
    int N = 100;

    std::ofstream ffile("cost.csv");
    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j <= N; ++j) {
            double x = xmin + (xmax - xmin) * i / N;
            double y = ymin + (ymax - ymin) * j / N;
            Eigen::Vector2d xv(x, y);
            ffile << x << "," << y << "," << f(xv, Q) << "\n";
        }
    }
    ffile.close();

    // ---- 2. Sample constraint c(x) = 0 ----
    std::ofstream cfile("constraint.csv");
    for (int i = 0; i <= N; ++i) {
        double x = xmin + (xmax - xmin) * i / N;
        double y = x*x + 2*x;  // from c(x)=0 → y = x² + 2x
        cfile << x << "," << y << "\n";
    }
    cfile.close();

    // ---- 3. Newton iterations ----
    Eigen::Vector3d xl;
    xl << -1.0, -1.0, 0.0;   // initial [x, y, λ]

    auto start_t = high_resolution_clock::now();
    int num_steps = 15;
    double rho = 10.0;
    std::ofstream nfile("newton.csv");

    for (int i = 0; i < num_steps; ++i) {
        Eigen::Vector2d x = xl.head<2>();
        double l = xl(2);
        nfile << x(0) << "," << x(1) << "\n";
        // xl = gauss_newton_step(xl, Q);
        
        Eigen::Vector3d dxl = gauss_newton_step(xl, Q);
        Eigen::Vector2d dx = dxl.head<2>();
        double dl = dxl(2);

        // Line search
        double alpha = 1.0;
        while (p(x + alpha * dx, Q, l + alpha * dl, rho) > p(x, Q, l, rho) + 0.01 * alpha * dp(x, Q, l, rho).dot(dxl))
            alpha *= 0.5;
        
        xl += alpha * dxl;
    }
    auto end_t = high_resolution_clock::now();
    std::cout << "Time Taken: " << duration_cast<microseconds>(end_t - start_t).count() << " us\n";
    nfile.close();

    std::cout << "Wrote: cost.csv, constraint.csv, newton.csv\n";
    return 0;
}
