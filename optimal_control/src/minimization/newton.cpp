#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <vector>
using namespace std::chrono;

// Use Newton's method to solve: min_x f(x) = x^4 + x^3 - x^2 - x

double f(double x) {
    return std::pow(x, 4) + std::pow(x, 3) - std::pow(x, 2) - x;
}

double df(double x) {
    return 4.0*std::pow(x,3) + 3.0 * std::pow(x,2) - 2.0 * x - 1.0;
}

double ddf(double x) {
    return 12.0 * std::pow(x,2) + 6.0 * x - 2.0;
}

double newton_step(double xp) {
    return xp - df(xp) / ddf(xp);
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
    double x = 0.3;
    int num_steps = 6;

    std::ofstream nfile("newton.csv");
    for (int i = 0; i < num_steps; ++i) {
        nfile << x << "," << f(x) << "\n";
        x = newton_step(x);
    }
    nfile.close();

    std::cout << "Data written to function.csv and newton.csv\n";


}