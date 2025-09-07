#include "rs.h"
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std::chrono;

int main(int argc, char* argv[]) {
    auto start = high_resolution_clock::now();
    double x1 = 1.0, y1 = 1.0, t1 = 0.0;
    double x2 = 2.0, y2 = 2.0, t2 = RS::MPI/2;
    double rad_curv = 5;

    // Get the min length RS curv
    int n;
    double t,u,v;
    double length = RS::min_length_rs(x1, y1, t1, x2, y2, t2, &n, &t, &u, &v, rad_curv);
    std::cout << "Length: " << length << " n: " << n << "\n";
    std::cout << "t: " << t << " u: " << u << " v: " << v << "\n";
    // Densify the min length RS curv
    double delta = 0.1;
    int num_points = (int) (length / delta) + 2;
    double path_x[num_points], path_y[num_points], path_t[num_points];
    path_x[0] = x1; path_y[0] = y1; path_t[0] = t1;
    int num_valid_points = RS::constRS(n, t, u, v, x1, y1, t1, delta, path_x, path_y, path_t, rad_curv);

    std::cout << "Path: n_points " << num_points << " num_valid_points " << num_valid_points << "\n";
    for (int i=0; i < num_valid_points; i++) 
        std::cout << path_x[i] << ", " << path_y[i] << ", " << path_t[i] << "\n";
    auto end = high_resolution_clock::now();
    std::cout << "Time Required: " << duration_cast<microseconds>(end - start).count() << " us\n"; 
    
    std::ofstream out_file("../assets/rs_path.txt");
    if (out_file.is_open()) {
        for (int i = 0; i < num_valid_points; ++i) {
            out_file << path_x[i] << " " << path_y[i] << " " << path_t[i] << "\n";
        }
        out_file.close();
        std::cout << "RS path saved to rs_path.txt\n";
    } else {
        std::cerr << "Failed to open file for writing.\n";
    }

    return 0;
}
