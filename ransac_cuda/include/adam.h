#ifndef ADAM_H
#define ADAM_H

#include <cmath>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace ransac {

typedef float F;

/**
 * @brief Container for model parameters and their gradients.
 *
 * This class manages a contiguous block of memory storing both the
 * parameter values and their corresponding gradients. It allocates
 * arrays of size `size` for `data` (parameters) and `grads`
 * (gradients), and frees them on destruction.
 *
 * It acts as a lightweight tensor-like structure for optimizers such
 * as Adam, with no additional functionality beyond memory management.
 */
class Params {
public:
    /**
     * @brief Construct a parameter container.
     *
     * Allocates memory for both parameter values and their gradients.
     *
     * @param size Number of parameters in the array.
     */
    Params(size_t size);

    /**
     * @brief Copy parameter values and gradients into this container.
     *
     * Replaces the internal `data` and `grads` arrays with the contents of
     * the provided buffers. Both input pointers must point to arrays of
     * length equal to `size`. The data is copied using `memcpy`, so the
     * input arrays must contain trivially copyable types (e.g., float/double).
     *
     * @param data_  Pointer to an array containing new parameter values.
     * @param grads_ Pointer to an array containing new gradients.
     *
     */
    void set(F* data_, F* grads_);

    /**
     * @brief Copy container values and gradients into arguments.

     * @param data_  Pointer to an array to be filled with parameter values.
     * @param grads_ Pointer to an array to be filled with gradients.
     *
     */
    void get(F* data_, F* grads_);

    /**
     * @brief Destructor.
     *
     * Frees the allocated parameter and gradient arrays.
     */
    ~Params();

    size_t size;        ///< Number of parameters stored.
    F* data{nullptr};   ///< Array of parameter values.
    F* grads{nullptr};  ///< Array of gradients corresponding to `data`.
};

/**
 * @brief Hyperparameters for the Adam optimizer.
 *
 * This struct stores the tunable parameters used by the Adam update rule:
 *  - lr:     Learning rate (step size).
 *  - beta1:  Exponential decay rate for the first moment (mean of gradients).
 *  - beta2:  Exponential decay rate for the second moment (uncentered variance).
 *  - eps:    Small constant to prevent division by zero.
 *
 * Defaults match the values from the original Adam paper.
 */
struct AdamOptions {
    F lr = 1e-3f;
    F beta1 = 0.9f;
    F beta2 = 0.999f;
    F eps = 1e-8f;
};

/**
 * @brief Adam optimizer implementation.
 *
 * This class implements the Adam optimization algorithm for updating
 * parameter vectors based on their gradients. It maintains first-
 * and second-moment exponential moving averages (`m`, `v`) and applies
 * bias corrections as described in the Adam paper.
 *
 * The optimizer operates on raw pointers to arrays of type `F` (float/double),
 * making it lightweight and suitable for custom tensor backends.
 */
class Adam {
public:
    /**
     * @brief Construct an Adam optimizer for a parameter vector.
     *
     * Allocates internal buffers (`m` and `v`) according to the parameter size
     * and initializes them to zero.
     *
     * @param params_size Number of parameters to optimize.
     * @param opt         Adam hyperparameters (learning rate, betas, epsilon).
     *
     * @throws std::runtime_error If `params_size` is zero.
     */
    Adam(size_t params_size, AdamOptions opt);

    /**
     * @brief Reset optimizer state.
     *
     * Sets moment estimates (`m`, `v`) to zero and resets the time step `t`.
     * Useful when reusing the optimizer for a fresh training run.
     */
    void reset();

    /**
     * @brief Apply one Adam update step.
     *
     * Updates the given parameter array in place using the provided gradients.
     * Performs moment updates, bias corrections, and Adam parameter update rule.
     *
     * @param params Pointer to parameter array (size = params_size).
     * @param grads  Pointer to gradient array (size = params_size).
     */
    void update(Params& params);

    /**
     * @brief Destructor.
     *
     * Frees the dynamically allocated moment buffers.
     */
    ~Adam();

private:
    size_t params_size;  ///< Number of parameters this optimizer manages.
    AdamOptions opt;     ///< Adam hyperparameter configuration.
    F* m{nullptr};       ///< First-moment estimates (mean of gradients).
    F* v{nullptr};       ///< Second-moment estimates (uncentered variance).
    size_t t{0};         ///< Time step counter used for bias correction.
};

}  // namespace ransac

#endif