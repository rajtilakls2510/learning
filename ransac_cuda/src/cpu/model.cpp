#ifndef MODEL_CPP
#define MODEL_CPP

#include "model.h"

namespace ransac {

LinearRegressor::LinearRegressor(size_t params_size) { params = new Params(params_size); }

Params* LinearRegressor::getParams() { return params; }

void LinearRegressor::output(F* input, size_t num_inputs, F* outputs) {
    if (input && num_inputs && outputs) {
        for (size_t i = 0; i < num_inputs; i++) {
            outputs[i] = 0;
            for (size_t pi = 0; pi + 1 < params->size; pi++)
                outputs[i] += input[params->size * i + pi] * params->data[pi];
            outputs[i] += params->data[params->size - 1];
        }
    } else {
        throw std::runtime_error("LinearRegression output: Either input/output/num_inputs invalid");
    }
}

F LinearRegressor::loss(F* input, size_t num_inputs) {
    F l = 0.0f;
    if (input && num_inputs > 0) {
        F* pred_outputs = new F[num_inputs];
        try {
            output(input, num_inputs, pred_outputs);
        } catch (const std::exception& e) {
            delete[] pred_outputs;
            throw e;
        }

        for (size_t i = 0; i < num_inputs; i++)
            l += pow(input[params->size * i + params->size - 1] - pred_outputs[i], 2.0f);
        l /= num_inputs;

        delete[] pred_outputs;
    } else {
        throw std::runtime_error("LinearRegression loss: Either input/num_inputs invalid");
    }
    return l;
}

// Assumes input shape: [num_inputs, params_size]
void LinearRegressor::computeAndStoreGrad(F* input, size_t num_inputs) {
    if (input && num_inputs > 0) {
        // Clear grads
        // for (size_t pi = 0; pi < params->size; pi++) params->grads[pi] = 0;
        params->clearGrads();

        F* pred_outputs = new F[num_inputs];
        try {
            output(input, num_inputs, pred_outputs);
        } catch (const std::exception& e) {
            delete[] pred_outputs;
            throw e;
        }

        // Calculate grads
        for (size_t inp = 0; inp < num_inputs; inp++) {
            // z_i - w_1 * x_i - w_2 * y_i - w_3
            F err = input[params->size * inp + params->size - 1] - pred_outputs[inp];

            for (size_t pi = 0; pi + 1 < params->size; pi++)
                params->grads[pi] -= 2.0f * err * input[params->size * inp + pi] / num_inputs;
            params->grads[params->size - 1] -= 2.0f * err / num_inputs;
        }

        delete[] pred_outputs;
    } else {
        throw std::runtime_error(
                "LinearRegression computeAndStoreGrad: Either input/num_inputs invalid");
    }
}

// Input shape: [num_inputs, params_size] Outputs: least loss
F LinearRegressor::fit(F* input, size_t num_inputs, F lr, size_t max_iterations, F min_loss) {
    params->clearData();
    params->clearGrads();

    // Initialize optimizer
    AdamOptions opt;
    opt.lr = lr;
    Adam adam(params->size, opt);

    F loss_ = 0.0f;
    for (size_t it = 0; it < max_iterations; it++) {
        // Compute gradients
        computeAndStoreGrad(input, num_inputs);
        // Step optimizer
        adam.update(*params);
        // Calculate loss
        loss_ = loss(input, num_inputs);
        if (loss_ < min_loss) break;
    }
    return loss_;
}

LinearRegressor::~LinearRegressor() {
    delete params;
    params = nullptr;
}

}  // namespace ransac

#endif