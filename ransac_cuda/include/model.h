#ifndef MODEL_H
#define MODEL_H

#include "adam.h"

namespace ransac {

/**
 * @class LinearRegressor
 * @brief Simple linear regression model with learnable parameters.
 *
 * This class represents a linear model of the form:
 * \f[
 * y = w^\top x
 * \f]
 * where `w` are the learnable parameters stored inside a `Params` object.
 *
 * The input format for all methods follows:
 *   - Each data sample has `params_size` features **plus one target value**
 *   - Total input tensor shape: `[num_inputs, params_size + 1]`
 *
 * The last element of each sample is treated as the ground-truth target.
 */
class LinearRegressor {
public:
    /**
     * @brief Construct a linear regressor.
     *
     * Allocates a Params object of size `params_size`, initialized to zeros.
     *
     * @param params_size Number of trainable parameters (feature count).
     */
    LinearRegressor(size_t params_size);

    /**
     * @brief Get the underlying parameter container.
     *
     * @return Pointer to the Params object holding model weights.
     */
    Params* getParams();

    /**
     * @brief Compute model outputs for a batch of input samples.
     *
     * Each input sample contains:
     *  - `params_size - 1` feature values
     *  - 1 target value (ignored in this function)
     *
     * Input layout:
     *   `[num_inputs, params_size]`
     *
     * Output layout:
     *   `[num_inputs]`
     *
     * @param input   Pointer to the input buffer.
     * @param num_inputs Number of samples.
     * @param outputs Pointer to array where outputs will be written.
     *
     * @throws std::runtime_error If input/output pointers are null or num_inputs is zero.
     */
    void output(F* input, size_t num_inputs, F* outputs);

    /**
     * @brief Compute mean squared error loss over a batch of samples.
     *
     * Input layout:
     *   `[num_inputs, params_size]`
     *
     * For each sample:
     *   - First `params_size - 1` entries are features.
     *   - Last entry is the target value.
     *
     * Loss computed as:
     * \f[
     * \frac{1}{N} \sum_i (y_i - \hat{y}_i)^2
     * \f]
     *
     * @param input Pointer to input data including targets.
     * @param num_inputs Number of samples.
     *
     * @return Mean squared error (MSE).
     *
     * @throws std::runtime_error If input pointer is null or num_inputs is zero.
     */
    F loss(F* input, size_t num_inputs);

    /**
     * @brief Trains the linear regressor using Adam optimization.
     *
     * Runs gradient descent with Adam on the given dataset until either
     * `max_iterations` is reached or the loss drops below `min_loss`.
     * Each input row is of shape [params_size], with the last element as the target.
     *
     * @param input          Training data buffer of size num_inputs * params_size.
     * @param num_inputs     Number of samples.
     * @param lr             Learning rate (default 0.5f).
     * @param max_iterations Maximum training steps (default 1000).
     * @param min_loss       Early stopping threshold (default 1e-9f).
     *
     * @return Final loss value.
     */
    F fit(F* input,
          size_t num_inputs,
          F lr = 0.5f,
          size_t max_iterations = 1000,
          F min_loss = 1e-9f);

    /**
     * @brief Destructor.
     *
     * Frees the allocated Params object.
     */
    ~LinearRegressor();

private:
    Params* params{nullptr};  ///< Trainable model parameters.

    /**
     * @brief Compute and store gradients of the MSE loss w.r.t. model parameters.
     *
     * This function assumes that each input sample has the layout:
     *   [x_0, x_1, ..., x_{P-1}, target]
     * where P = params->size.
     *
     * For each sample i, the prediction is:
     *    y_i = sum_j ( w_j * x_{i,j} )
     *
     * The loss is:
     *    L = (1/N) * sum_i (y_i - target_i)^2
     *
     * This function computes:
     *    dL/dw_j  for all j
     *
     * and stores the results in `params->grads`.
     *
     * @param input Pointer to input samples in row-major format.
     * @param num_inputs Number of samples.
     *
     * @throws std::runtime_error If input is null or num_inputs == 0.
     */

    void computeAndStoreGrad(F* input, size_t num_inputs);
};

}  // namespace ransac

#endif