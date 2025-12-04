#ifndef ADAM_CU
#define ADAM_CU

#include <cuda_runtime.h>
#include <stdio.h>

#include "adam.h"

namespace ransac {

#define CUDA_CHECK(ans)                       \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDAassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void adam_update_kernel(
        F* params,
        F* grads,
        F* m,
        F* v,
        F lr,
        F beta1,
        F beta2,
        F bias_correction1,
        F bias_correction2,
        F eps,
        int t,
        int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
        F m_hat = m[i] / bias_correction1;
        F v_hat = v[i] / bias_correction2;
        params[i] = params[i] - lr * m_hat / (eps + sqrt(v_hat));
    }
}

Adam::Adam(size_t params_size, AdamOptions opt) : params_size(params_size), opt(opt) {
    if (params_size < 1)
        throw std::runtime_error("Adam: number of params must be greater than zero");
    std::cout << "Using CUDA version of Adam\n";
    CUDA_CHECK(cudaMalloc(&m, params_size * sizeof(F)));
    CUDA_CHECK(cudaMalloc(&v, params_size * sizeof(F)));
    reset();
}

void Adam::reset() {
    CUDA_CHECK(cudaMemset(m, 0, params_size * sizeof(F)));
    CUDA_CHECK(cudaMemset(v, 0, params_size * sizeof(F)));
    t = 0;
}

void Adam::update(Params& params) {
    t++;
    F bias_correction1 = 1.0 - pow(opt.beta1, (F)t);
    F bias_correction2 = 1.0 - pow(opt.beta2, (F)t);

    int tpb = params_size < 1024 ? params_size : 1024;
    int bpd = (params_size + tpb - 1) / tpb;
    adam_update_kernel<<<bpd, tpb>>>(
            params.data,
            params.grads,
            m,
            v,
            opt.lr,
            opt.beta1,
            opt.beta2,
            bias_correction1,
            bias_correction2,
            opt.eps,
            t,
            params_size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

Adam::~Adam() {
    CUDA_CHECK(cudaFree(m));
    CUDA_CHECK(cudaFree(v));
}

Params::Params(size_t size) : size(size) {
    CUDA_CHECK(cudaMalloc(&data, size * sizeof(F)));
    CUDA_CHECK(cudaMalloc(&grads, size * sizeof(F)));

    CUDA_CHECK(cudaMemset(data, 0, size * sizeof(F)));
    CUDA_CHECK(cudaMemset(grads, 0, size * sizeof(F)));
}

void Params::set(F* data_, F* grads_) {
    if (data_) CUDA_CHECK(cudaMemcpy(data, data_, size * sizeof(F), cudaMemcpyHostToDevice));
    if (grads_) CUDA_CHECK(cudaMemcpy(grads, grads_, size * sizeof(F), cudaMemcpyHostToDevice));
}

void Params::get(F* data_, F* grads_) {
    if (data_) CUDA_CHECK(cudaMemcpy(data_, data, size * sizeof(F), cudaMemcpyDeviceToHost));
    if (grads_) CUDA_CHECK(cudaMemcpy(grads_, grads, size * sizeof(F), cudaMemcpyDeviceToHost));
}

Params::~Params() {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(grads));
}

}  // namespace ransac
#endif