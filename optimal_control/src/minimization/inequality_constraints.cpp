#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
using namespace std;
using namespace std::chrono;

// Solve min_x 0.5 * (x - [1; 0])' @ Q @ (x - [1; 0]) s.t. [1.0 -1.0] @ x  <= -1.0 using:
//  - Augmented Lagrangian

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

double c(const Eigen::Vector2d& x, const Eigen::Vector2d& A /* should be a matrix */, const double b) {
    // A @ x - b
    return A.dot(x) - b;
}

Eigen::Vector2d dc(const Eigen::Vector2d& x, const Eigen::Vector2d& A /* should be a matrix */, const double b) {
    return A;
}

double AL(const Eigen::Vector2d& x, const double lamda, const double rho, const Eigen::Matrix2d& Q, const Eigen::Vector2d& A, const double b) {
    double p = max(0.0, c(x, A, b));
    return f(x, Q) + lamda * p + (rho/2) * (p*p);    // TODO for multi dim: Second term lamda^T @ p, Third term must be p^T @ p
}

bool is_pos_def(const Eigen::Matrix2d& H) {
    Eigen::LLT<Eigen::Matrix2d> llt(H);
    return llt.info() == Eigen::Success;
}

Eigen::Vector2d newton_solve_al(const Eigen::Vector2d& x0, const Eigen::Matrix2d& Q, const Eigen::Vector2d& A, const double b, double lamda, const double rho) {
    Eigen::Vector2d x = x0;
    double p = max(0.0, c(x, A, b));
    Eigen::Vector2d C;  /* TODO: For nd, C = zeros(1,n) */
    C.setZero();
    if (c(x, A, b) >= 0 /* TODO: For higher dimensions, only c(x) >= 0 must be set*/) {
        C = dc(x,A,b);
    } 

    Eigen::Vector2d g = df(x, Q) + (lamda + rho * p) * C;   /* TODO: For nd (lamda + rho * p) @ C^T */
    double beta = 1e-6;
    Eigen::Matrix2d I = Eigen::Matrix2d::Identity();
    while (g.norm() >= 1e-8) {
        Eigen::Matrix2d H = ddf(x,Q) + C * C.transpose();
        
        // Regularize until PD
        while (!is_pos_def(H)) {
            H += beta * I;
            beta *= 10.0;
        }

        Eigen::Vector2d dx = -H.ldlt().solve(g); 

        // ==== Line Search here doesn't work for some reason
        // // Backtracking line search (Armijo)
        double alpha = 1.0;
        // double b_ = 1e-4;
        // double c_ = 0.5;

        // while (f(x + alpha*dx, Q) > f(x, Q) + b_ * alpha * (df(x,Q)).dot(dx)) {
        //     alpha *= c_;
        // }

        x = x + alpha * dx; 

        p = max(0.0, c(x, A, b));
        C.setZero();
        if (c(x, A, b) >= 0 /* TODO: For higher dimensions, only c(x) >= 0 must be set*/) {
            C = dc(x,A,b);
        } 
        g = df(x, Q) + (lamda + rho * p) * C;
    }
    return x;
}

int main() {

    Eigen::Matrix2d Q;
    Q << 0.5, 0.0,
         0.0, 1.0;

    Eigen::Vector2d A;
    A << 1.0, -1.0;
    double b = -1.0;

    // ---------- cost grid ----------
    double xmin = -2.0, xmax = 2.0;
    double ymin = -2.0, ymax = 2.0;
    int N = 120;

    std::ofstream costfile("cost.csv");
    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j <= N; ++j) {
            double x = xmin + (xmax - xmin) * i / N;
            double y = ymin + (ymax - ymin) * j / N;
            Eigen::Vector2d xv(x, y);
            costfile << x << "," << y << "," << f(xv, Q) << "\n";
        }
    }
    costfile.close();

    // ---------- constraint boundary ----------
    std::ofstream cfile("constraint.csv");
    for (int i = 0; i <= N; ++i) {
        double x = xmin + (xmax - xmin) * i / N;
        double y = x + 1.0;   // from x - y = -1 → y = x + 1
        cfile << x << "," << y << "\n";
    }
    cfile.close();

    // ---------- Augmented Lagrangian iterations ----------
    Eigen::Vector2d x;
    x << -3.0, 2.0;

    double lambda = 0.0;
    double rho = 1.0;

    std::ofstream nfile("newton.csv");

    int num_steps = 15;
    auto start_t = high_resolution_clock::now();
    for (int k = 0; k < num_steps; ++k) {
        nfile << x(0) << "," << x(1) << "\n";
        std::cout << "\rk: " << k << " / " << num_steps << "\n";
        x = newton_solve_al(x, Q, A, b, lambda, rho);

        double cx = c(x, A, b);
        lambda = std::max(0.0, lambda + rho * cx);
        // rho = 10.0 * rho;    // This doesn't work for some reason
    }
    auto end_t = high_resolution_clock::now();
    std::cout << "Total time: " << duration_cast<microseconds>(end_t - start_t).count() << " us\n";
    nfile.close();

    std::cout << "Wrote cost.csv, constraint.csv, newton.csv\n";
    return 0;
}
