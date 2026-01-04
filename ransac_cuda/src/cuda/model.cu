#ifndef MODEL_CU
#define MODEL_CU

#include "common.hpp"
#include "model.h"

namespace ransac {

__global__ void output_kernel(
        F* input, F* outputs, F* params, size_t params_size, size_t num_inputs) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_inputs) {
        outputs[i] = 0;
        for (size_t pi = 0; pi + 1 < params_size; pi++)
            outputs[i] += input[params_size * i + pi] * params[pi];
        outputs[i] += params[params_size - 1];
    }
}

__global__ void mse_loss_kernel(
        F* input,
        F* pred_output,
        F* partial_sums_per_block,
        size_t params_size,
        size_t num_inputs) {
    extern __shared__ F sdata[];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + tid;

    // Calculate loss for each thread
    F val = 0.0f;
    if (i < num_inputs) val = pow(input[params_size * i + params_size - 1] - pred_output[i], 2.0f);

    sdata[tid] = val;
    __syncthreads();

    // Reduce to put values per dimension
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partial_sums_per_block[blockIdx.x] = sdata[0];
}

__global__ void compute_grad_kernel(
        F* input, F* pred_outputs, F* grads, size_t params_size, size_t num_inputs) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_inputs) {
        F err = input[params_size * i + params_size - 1] - pred_outputs[i];
        for (size_t pi = 0; pi + 1 < params_size; pi++)
            grads[pi] -= 2.0f * err * input[params_size * i + pi] / num_inputs;
        grads[params_size - 1] -= 2.0f * err / num_inputs;
    }
}

LinearRegressor::LinearRegressor(size_t params_size) { params = new Params(params_size); }

Params* LinearRegressor::getParams() { return params; }

void LinearRegressor::output(F* input, size_t num_inputs, F* outputs) {
    if (input && num_inputs > 0 && outputs) {
        int tpb = num_inputs < 1024 ? num_inputs : 1024;
        int bpd = (num_inputs + tpb - 1) / tpb;
        output_kernel<<<bpd, tpb>>>(input, outputs, params->data, params->size, num_inputs);
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        throw std::runtime_error("LinearRegression output: Either input/output/num_inputs invalid");
    }
}

F LinearRegressor::loss(F* input, size_t num_inputs) {
    F l = 0.0f;
    if (input && num_inputs > 0) {
        F* pred_outputs;
        CUDA_CHECK(cudaMalloc(&pred_outputs, num_inputs * sizeof(F)));
        try {
            output(input, num_inputs, pred_outputs);
        } catch (const std::exception& e) {
            CUDA_CHECK(cudaFree(pred_outputs));
            throw e;
        }

        int tpb = num_inputs < 1024 ? num_inputs : 1024;
        int bpd = (num_inputs + tpb - 1) / tpb;

        F* d_partial;
        CUDA_CHECK(cudaMalloc(&d_partial, bpd * sizeof(F)));
        mse_loss_kernel<<<bpd, tpb, tpb * sizeof(F)>>>(
                input, pred_outputs, d_partial, params->size, num_inputs);
        CUDA_CHECK(cudaDeviceSynchronize());

        F* h_partial = new F[bpd];
        CUDA_CHECK(cudaMemcpy(h_partial, d_partial, bpd * sizeof(F), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_partial));
        CUDA_CHECK(cudaFree(pred_outputs));

        // Final CPU reduction
        for (size_t i = 0; i < bpd; i++) l += h_partial[i];
        l /= num_inputs;

        delete[] h_partial;
        h_partial = nullptr;
    } else {
        throw std::runtime_error("LinearRegression loss: Either input/num_inputs invalid");
    }
    return l;
}

void LinearRegressor::computeAndStoreGrad(F* input, size_t num_inputs) {
    if (input && num_inputs > 0) {
        params->clearGrads();

        F* pred_outputs;
        CUDA_CHECK(cudaMalloc(&pred_outputs, num_inputs * sizeof(F)));
        try {
            output(input, num_inputs, pred_outputs);
        } catch (const std::exception& e) {
            CUDA_CHECK(cudaFree(pred_outputs));
            throw e;
        }

        // Calculate grads
        int tpb = num_inputs < 1024 ? num_inputs : 1024;
        int bpd = (num_inputs + tpb - 1) / tpb;
        compute_grad_kernel<<<bpd, tpb>>>(
                input, pred_outputs, params->grads, params->size, num_inputs);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaFree(pred_outputs));

    } else {
        throw std::runtime_error(
                "LinearRegression computeAndStoreGrad: Either input/num_inputs invalid");
    }
}

F LinearRegressor::fit(F* input, size_t num_inputs, F lr, size_t max_iterations, F min_loss) {
    params->clearData();
    params->clearGrads();

    // Initialize optimizer
    AdamOptions opt;
    opt.lr = lr;
    Adam adam(params->size, opt);

    F* d_input;
    CUDA_CHECK(cudaMalloc(&d_input, num_inputs * params->size * sizeof(F)));
    CUDA_CHECK(cudaMemcpy(
            d_input, input, num_inputs * params->size * sizeof(F), cudaMemcpyHostToDevice));

    F loss_ = 0.0f;
    for (size_t it = 0; it < max_iterations; it++) {
        // Compute gradients
        computeAndStoreGrad(d_input, num_inputs);
        // Step optimizer
        adam.update(*params);
        // Calculate loss
        loss_ = loss(d_input, num_inputs);
        if (loss_ < min_loss) break;
    }
    CUDA_CHECK(cudaFree(d_input));

    return loss_;
}

LinearRegressor::~LinearRegressor() {
    delete params;
    params = nullptr;
}

}  // namespace ransac

#endif