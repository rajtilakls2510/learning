#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
using namespace std;
using namespace std::chrono;

/**
 * Solve min_x 0.1*(1 - x_1)^2 + 0.001*(x_2 - x_1^2)^2 + 0.1*sin(3x_1)*sin(3x_2)
 *      such that   sin(1.75*x_1) + x_2^2 - 1.0 = 0
 *                  x_1*x_2 - 0.25 = 0
 *                  x_1^2 + x_2^2 - 2 <= 0
 *                  x_2 - 0.1 <= 0
 * 
 * Use an Augmented Lagrangian Method with Gauss-Newton to solve inner optimization problem
 */

// Cost
double f(const Eigen::VectorXd& x) {
    return 0.1 * (1 - pow(x(0), 2)) + 0.001 * pow(x(1) - pow(x(0), 2), 2) + 0.1 * sin(3 * x(0)) * sin(3 * x(1));
} 

// gradient of cost
Eigen::VectorXd dfdx(const Eigen::VectorXd& x) {
    Eigen::VectorXd g(2);
    g << -0.2*(1-x(0)) - 0.004*(x(1)-pow(x(0),2))*x(0) + 0.3*cos(3*x(0))*sin(3*x(1)), 0.002*(x(1) - pow(x(0),2)) + 0.3*sin(3*x(0))*cos(3*x(1));
    return g;
}

// hessian of cost
Eigen::MatrixXd d2fdx2(const Eigen::VectorXd& x) {
    Eigen::MatrixXd h(2,2);
    h << 0.2 - 0.004*(x(1)-pow(x(0),2)) + 0.008*pow(x(0),2) - 0.9*sin(3*x(0))*sin(3*x(1)), -0.004*x(0) + 0.9*cos(3*x(0))*cos(3*x(1)),
        -0.004*x(0) + 0.9*cos(3*x(0))*cos(3*x(1)), 0.002 - 0.9*sin(3*x(0))*sin(3*x(1));
    return h;
}

// Equality constraints
Eigen::VectorXd h(const Eigen::VectorXd& x) {
    Eigen::VectorXd h_(2);
    h_ << sin(1.75*x(0)) + pow(x(1), 2) - 1.0, x(0) * x(1) - 0.25;
    return h_;
}

// Jacobian of equality
Eigen::MatrixXd dhdx(const Eigen::VectorXd& x) {
    Eigen::MatrixXd j(2,2);
    j << 1.75*cos(1.75*x(0)), 2*x(1),
         x(1), x(0);
    return j;
}

// Inequality constraints
Eigen::VectorXd g(const Eigen::VectorXd& x) {
    Eigen::VectorXd g_(2);
    g_ << pow(x(0), 2) + pow(x(1), 2) - 2, -0.1 + x(1);
    return g_;
}

// Jacobian of inequality
Eigen::MatrixXd dgdx(const Eigen::VectorXd& x) {
    Eigen::MatrixXd j(2,2);
    j << 2*x(0), 2*x(1),
        0, 1;
    return j;
}

// Gradient of Lagrangian wrt x
Eigen::VectorXd dldx(const Eigen::VectorXd& x, const Eigen::VectorXd& n, const Eigen::VectorXd& lamda, const double rho) {
    Eigen::MatrixXd dh = dhdx(x);
    Eigen::VectorXd C(2);
    C.setZero();
    for (int i=0; i<2; i++) {
        double l = lamda(i) + rho * g(x)(i);
        if (l > 0)
            C(i) = l;
    }
    // std::cout << "C_ dldx: " << C << "\n";
    return dfdx(x) + dh.transpose() * (n + rho * h(x)) + dgdx(x).transpose() * C; 
}

// Checks whether given matrix is positive definite
bool is_pos_def(const Eigen::MatrixXd& H) {
    Eigen::LLT<Eigen::MatrixXd> llt(H);
    return llt.info() == Eigen::Success;
}

// Augmented Lagrangian
double al(const Eigen::VectorXd& x, const Eigen::VectorXd& n, const Eigen::VectorXd& lamda, const double rho) {
    Eigen::VectorXd C(2);
    C.setZero();
    for (int i=0; i<2; i++) {
        double l = lamda(i) + rho * g(x)(i);
        if (l > 0)
            C(i) = pow(l, 2);
    }
    double al_ = f(x) + n.transpose() * h(x) + 0.5 * rho * h(x).norm() + 0.5 / rho * (C.sum() - lamda.squaredNorm());
    return al_;
}

// Solve AL subproblem
Eigen::VectorXd solve_al_subproblem(const Eigen::VectorXd& x_0, const Eigen::VectorXd& n, const Eigen::VectorXd& lamda, const double rho) {
    Eigen::VectorXd x = x_0;
    // std::cout << "x_0: " << x_0 << "\n";
    Eigen::VectorXd g = dldx(x, n, lamda, rho);
    // std::cout << "g: " << g << "\n";
    double prev_g_norm = INFINITY;

    // while (g.norm() > 1e-8 && abs(g.norm() - prev_g_norm) > 1e-8) {
    while (abs(g.norm() - prev_g_norm) > 1e-8) {
        prev_g_norm = g.norm();
        // std::cout << "g.norm(): " << g.norm() << "\n";
        // Find Active inequality constraints
        Eigen::MatrixXd C(2,2);
        C.setZero();
        Eigen::MatrixXd g_ = dgdx(x);
        for (int i=0; i<2; i++) {
            double l = lamda(i) + rho * g(x)(i);
            // Copy row i of dgdx into row i of C
            if (l > 0)
                C.row(i) = g_.row(i);
                // C.block<2>(i) = g_(i);
        }

        // std::cout << "C: " << C << "\n";

        // Compute Hessian
        Eigen::MatrixXd H = d2fdx2(x) + rho * dhdx(x).transpose() * dhdx(x) + rho * C.transpose() * C;

        // std::cout << "H: " << H << "\n";

        // Regularize until PD
        double beta = 1e-6;
        Eigen::MatrixXd I = Eigen::Matrix2d::Identity();
        while (!is_pos_def(H)) {
            H += beta * I;
            beta *= 10.0;
            // std::cout << "beta: " << beta << "\n";
        }

        // std::cout << "Regularized H: " << H << "\n";

        // Solve KKT
        Eigen::VectorXd dx = - H.ldlt().solve(g);

        // std::cout << "dx: " << dx << "\n";

        // Backtracking line search using merit function
        double alpha = 1.0;
        
        while (al(x + alpha * dx, n, lamda, rho) > al(x, n, lamda, rho) + 0.01 * alpha * g.transpose() * dx ){//&& alpha > 1e-2) {
            // std::cout << "al dx: " << al(x + alpha * dx, n, lamda, rho) << "\n";
            // std::cout << "al : " << al(x, n, lamda, rho) + 0.01 * alpha * g.transpose() * dx  << "\n";
            alpha *= 0.5;
            // std::cout << "alpha: " << alpha << "\n";
        }
        
        // Update x
        x = x + alpha * dx;
        // std::cout << "x: " << x << "\n";

        // Compute gradient
        g = dldx(x, n, lamda, rho);
        // std::cout << "g: " << g << "\n";
    }
    return x;
}

void al_step(Eigen::VectorXd& x, Eigen::VectorXd& n, Eigen::VectorXd& lamda, const double rho) {
    // std::cout << "Solving Al Subproblem\n";
    x = solve_al_subproblem(x, n,lamda, rho);
    // std::cout << "Updating new\n";
    n = n + rho * h(x);
    // std::cout << "n: " << n << "\n";
    // std::cout << "Updating Lambda\n";
    Eigen::VectorXd C(2);
    C.setZero();
    for (int i=0; i<2; i++) {
        double l = lamda(i) + rho * g(x)(i);
        if (l > 0)
            C(i) = l;
    }
    // std::cout << "CL: " << C << "\n";
    lamda = C;
    // std::cout << "lamda: " << lamda << "\n";

}

int main(int argc, char* argv[]) {

    using Vec = Eigen::VectorXd;

    // -----------------------------
    // 1. Sample cost & constraints
    // -----------------------------
    double xmin = -2.5, xmax = 2.5;
    double ymin = -2.5, ymax = 2.5;
    int N = 100;

    std::ofstream cost_file("cost.csv");
    std::ofstream eq_file("equalities.csv");
    std::ofstream ineq_file("inequalities.csv");

    for (int i = 0; i <= N; ++i) {
        for (int j = 0; j <= N; ++j) {
            double x1 = xmin + (xmax - xmin) * i / N;
            double x2 = ymin + (ymax - ymin) * j / N;

            Vec x(2);
            x << x1, x2;

            // Cost
            cost_file << x1 << "," << x2 << "," << f(x) << "\n";

            // Equalities
            Vec hv = h(x);
            eq_file << x1 << "," << x2 << "," << hv(0) << "," << hv(1) << "\n";

            // Inequalities
            Vec gv = g(x);
            ineq_file << x1 << "," << x2 << "," << gv(0) << "," << gv(1) << "\n";
        }
    }

    cost_file.close();
    eq_file.close();
    ineq_file.close();

    // -----------------------------
    // 2. Augmented Lagrangian steps
    // -----------------------------
    Vec x(2);
    x << -2.0, -2.0;

    Vec n(2);      // equality multipliers
    Vec lamda(2);  // inequality multipliers
    n.setZero();
    lamda.setZero();

    double rho = 1.0;

    std::ofstream al_file("al_steps.csv");

    int outer_iters = 100;
    auto start_total = high_resolution_clock::now();
    for (int k = 0; k < outer_iters; ++k) {
        auto start_t = high_resolution_clock::now();
        al_file << x(0) << "," << x(1) << "\n" << std::flush;
        std::cout << "k: " << k << " / " << outer_iters; 
        al_step(x, n, lamda, rho);
        // rho *= 1.1;
        auto end_t = high_resolution_clock::now();
        std::cout << " Time: " << duration_cast<microseconds>(end_t - start_t).count() << " us\n";
    }
    auto end_total = high_resolution_clock::now();
    std::cout << "Total time: " << duration_cast<milliseconds>(end_total - start_total).count() << " ms\n";
    al_file.close();

    std::cout << "Wrote: cost.csv, equalities.csv, inequalities.csv, al_steps.csv\n";
    return 0;
}