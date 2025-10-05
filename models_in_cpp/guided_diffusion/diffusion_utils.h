#ifndef DIFFUSION_UTILS_H
#define DIFFUSION_UTILS_H

#include <torch/torch.h>

namespace ddpm {

using namespace torch::indexing;

inline torch::Tensor linear_schedule(
        int timesteps, float beta_start = 1e-4, float beta_end = 1e-2) {
    return torch::linspace(beta_start, beta_end, timesteps);
}

inline torch::Tensor cosine_beta_schedule(int timesteps) {
    int steps = timesteps + 1;
    float s = 0.008;
    torch::Tensor x = torch::linspace(0, steps, steps);
    torch::Tensor alphas_cumprod =
            torch::pow(torch::cos(((x / steps) + s) / (1 + s) * M_PI * 0.5), 2);
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0];
    torch::Tensor betas =
            1 - (alphas_cumprod.index({Slice(1, None)}) / alphas_cumprod.index({Slice(0, -1)}));
    return torch::clip(betas, 0, 0.999);
}

inline torch::Tensor extract(torch::Tensor a, torch::Tensor t, const at::IntArrayRef& x_shape) {
    int64_t b = t.size(0);
    auto gathered = a.gather(-1, t);
    // Create target shape: (b, 1, 1, ..., 1) with (len(x_shape) - 1) ones
    std::vector<int64_t> target_shape = {b};
    target_shape.insert(target_shape.end(), x_shape.size() - 1, 1);
    return gathered.reshape(target_shape);
}

inline torch::Tensor approx_standard_normal_cdf(torch::Tensor x) {
    /*
    A fast approximation of the cumulative distribution function of the
    standard normal.
    */
    return 0.5 * (1.0 + torch::tanh(
                                torch::sqrt(torch::tensor(2.0 / M_PI)) *
                                (x + 0.044715 * torch::pow(x, 3))));
}

inline torch::Tensor discretized_gaussian_log_likelihood(
        torch::Tensor x, torch::Tensor means, torch::Tensor log_scales) {
    torch::Tensor centered_x = x - means;
    torch::Tensor inv_stdv = torch::exp(-log_scales);
    torch::Tensor plus_in = inv_stdv * (centered_x + 1.0 / 255.0);
    torch::Tensor cdf_plus = approx_standard_normal_cdf(plus_in);
    torch::Tensor min_in = inv_stdv * (centered_x - 1.0 / 255.0);
    torch::Tensor cdf_min = approx_standard_normal_cdf(min_in);
    torch::Tensor log_cdf_plus = torch::log(cdf_plus.clamp(1e-12));
    torch::Tensor log_one_minus_cdf_min = torch::log((1.0 - cdf_min).clamp(1e-12));
    torch::Tensor cdf_delta = cdf_plus - cdf_min;
    torch::Tensor log_probs = torch::where(
            x < -0.999,
            log_cdf_plus,
            torch::where(x > 0.999, log_one_minus_cdf_min, torch::log(cdf_delta.clamp(1e-12))));
    return log_probs;
}

}  // namespace ddpm

#endif