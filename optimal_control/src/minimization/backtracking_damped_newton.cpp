#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <vector>
using namespace std::chrono;

// Use Backtracking Damped Newton's method to solve: min_x f(x) = x^4 + x^3 - x^2 - x

double f(double x) {
    return std::pow(x, 4) + std::pow(x, 3) - std::pow(x, 2) - x;
}

double df(double x) {
    return 4.0*std::pow(x,3) + 3.0 * std::pow(x,2) - 2.0 * x - 1.0;
}

double ddf(double x) {
    return 12.0 * std::pow(x,2) + 6.0 * x - 2.0;
}

bool is_pos_def(double H) { // TODO: For higher-dimensional, switch to solver based
    return H > 0.0; 
}

double backtracking_regularized_newton_step(double xp) {
    double beta = 1.0;
    double H = ddf(xp);

    while (!is_pos_def(H)) {
        H = H + beta * 1.0; // TODO: For higher-dimensional, switch to Identity
    }

    double dx = -df(xp) / H;
    
    double b = 0.1;
    double c = 0.5;
    double alpha = 1.0;
    while (f(xp + alpha * dx) > f(xp) + alpha * b * df(xp) * dx) {
        alpha = c * alpha;
    }

    return xp + alpha * dx;
}

int main() {

     // ---- sample f(x) ----
    double xmin = -1.75, xmax = 1.25;
    int N = 1000;

    std::ofstream ffile("function.csv");
    for (int i = 0; i <= N; ++i) {
        double x = xmin + (xmax - xmin) * i / N;
        ffile << x << "," << f(x) << "\n";
    }
    ffile.close();

     // ---- Newton iterates ----
    double x = -0.25;
    int num_steps = 6;

    std::ofstream nfile("newton.csv");
    for (int i = 0; i < num_steps; ++i) {
        nfile << x << "," << f(x) << "\n";
        x = backtracking_regularized_newton_step(x);
    }
    nfile.close();

    std::cout << "Data written to function.csv and newton.csv\n";


}