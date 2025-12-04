#include "adam.h"

#include <cmath>
#include <iostream>

using namespace ransac;

// Loss function: f(x,y) = (x-5)^2 + (y+2)^2
F loss_function(const F* p) {
    F x = p[0];
    F y = p[1];
    return (x - 5.0f) * (x - 5.0f) + (y + 2.0f) * (y + 2.0f);
}

// Gradient: df/dx = 2(x-5), df/dy = 2(y+2)
void compute_grad(const F* p, F* g) {
    g[0] = 2.0f * (p[0] - 5.0f);
    g[1] = 2.0f * (p[1] + 2.0f);
}

int main() {
    AdamOptions opt;
    opt.lr = 0.1f;

    const size_t num_params = 2;
    F p[num_params] = {10.0f, -10.0f};  // Start far from optimum
    F g[num_params] = {0.0f, 0.0f};

    Params params(num_params);
    params.set(p, g);

    Adam adam(num_params, opt);

    std::cout << "Initial params: [" << p[0] << ", " << p[1] << "]\n";

    int num_iters = 300;
    for (int iter = 0; iter < num_iters; ++iter) {
        // Get latest param values
        params.get(p, g);

        // Compute gradients
        compute_grad(p, g);
        params.set(p, g);

        // Update with Adam
        adam.update(params);

        // Read updated params
        params.get(p, g);

        if (iter % 10 == 0) {
            std::cout << "Iter " << iter << "  p = [" << p[0] << ", " << p[1] << "]"
                      << "  loss = " << loss_function(p) << "\n";
        }
    }

    std::cout << "\nFinal params: [" << p[0] << ", " << p[1] << "]\n";
    std::cout << "True optimum is [5, -2]\n";

    return 0;
}
