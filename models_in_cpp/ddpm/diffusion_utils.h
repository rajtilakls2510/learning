#ifndef DIFFUSION_UTILS_H
#define DIFFUSION_UTILS_H

#include <torch/torch.h>

namespace ddpm {

using namespace torch::indexing;

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

}  // namespace ddpm

#endif