#include "model.h"

#include <chrono>
#include <vector>
using namespace ransac;
using namespace std::chrono;

// Create synthetic linear data:
//   z = 3*x + 2*y + 1
void generate_data(F* buffer, size_t num_samples, size_t param_size) {
    for (size_t i = 0; i < num_samples; i++) {
        F x = (F)i * 0.01f;
        F y = (F)i * 0.02f;
        F z = 3.0f * x + 2.0f * y + 1.0f;

        buffer[i * param_size + 0] = x;
        buffer[i * param_size + 1] = y;
        buffer[i * param_size + 2] = z;
    }
}
int main() {
    const size_t param_size = 3;  // w1, w2, b
    const size_t num_samples = 10000;

    F* data = new F[num_samples * param_size];
    generate_data(data, num_samples, param_size);

    LinearRegressor model(param_size);
    Params* p = model.getParams();
    F p_data[p->size];
    F p_grads[p->size];
    p->get(p_data, p_grads);

    std::cout << "Initial params: [" << p_data[0] << ", " << p_data[1] << ", " << p_data[2]
              << "]\n";

    auto start_t = high_resolution_clock::now();
    F loss = model.fit(data, num_samples, 0.5f, 10000, 1e-9f);
    auto end_t = high_resolution_clock::now();

    std::cout << "Time taken: " << duration_cast<milliseconds>(end_t - start_t).count() << " ms\n";
    std::cout << "Final loss: " << loss << "\n";

    p->get(p_data, p_grads);
    std::cout << "Final params: [" << p_data[0] << ", " << p_data[1] << ", " << p_data[2] << "]\n";

    std::cout << "Ground truth: [3, 2, 1]\n";
    delete[] data;
    return 0;
}