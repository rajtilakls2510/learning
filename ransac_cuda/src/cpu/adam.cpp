#ifndef ADAM_CPP
#define ADAM_CPP

#include "adam.h"

namespace ransac {

Adam::Adam(size_t params_size, AdamOptions opt) : params_size(params_size), opt(opt) {
    if (params_size < 1)
        throw std::runtime_error("Adam: number of params must be greater than zero");
    std::cout << "Using CPU version of Adam\n";
    m = new F[params_size];
    v = new F[params_size];
    reset();
}

void Adam::reset() {
    memset(m, 0, params_size * sizeof(F));
    memset(v, 0, params_size * sizeof(F));
    t = 0;
}

void Adam::update(Params& params) {
    t++;

    F bias_correction1 = 1.0 - pow(opt.beta1, (F)t);
    F bias_correction2 = 1.0 - pow(opt.beta2, (F)t);

    for (size_t pi = 0; pi < params_size; pi++) {
        F g = params.grads[pi];

        // Update biased moments
        m[pi] = opt.beta1 * m[pi] + (1.0 - opt.beta1) * g;
        v[pi] = opt.beta2 * v[pi] + (1.0 - opt.beta2) * g * g;

        // Bias-corrected estimates
        F m_hat = m[pi] / bias_correction1;
        F v_hat = v[pi] / bias_correction2;

        // Parameter update
        params.data[pi] -= opt.lr * m_hat / (std::sqrt(v_hat) + opt.eps);
    }
}

Adam::~Adam() {
    delete[] m;
    delete[] v;
}

Params::Params(size_t size) : size(size) {
    data = new F[size];
    grads = new F[size];
    clearData();
    clearGrads();
}

void Params::set(F* data_, F* grads_) {
    if (data_) std::memcpy(data, data_, size * sizeof(F));
    if (grads_) std::memcpy(grads, grads_, size * sizeof(F));
}

void Params::get(F* data_, F* grads_) {
    if (data_) std::memcpy(data_, data, size * sizeof(F));
    if (grads_) std::memcpy(grads_, grads, size * sizeof(F));
}

void Params::clearData() { memset(data, 0, size * sizeof(F)); }
void Params::clearGrads() { memset(grads, 0, size * sizeof(F)); }

Params::~Params() {
    delete[] data;
    delete[] grads;
}

}  // namespace ransac

#endif