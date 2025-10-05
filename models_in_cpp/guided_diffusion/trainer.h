#ifndef TRAINER_H
#define TRAINER_H

#include <filesystem>
#include <string>

#include "diffusion_utils.h"
#include "mnist_loader.hpp"
#include "network.h"

namespace ddpm {
using namespace MNIST;

class Trainer {
public:
    Trainer(std::string data_path,
            std::string checkpoint_path,
            int max_diffusion_time,
            bool use_cpu = false);

    torch::Tensor predict_xstart_from_eps(torch::Tensor x_t, torch::Tensor t, torch::Tensor noise);
    torch::Tensor q_sample(torch::Tensor x_start, torch::Tensor t, torch::Tensor noise);
    void train_step(Batch batch, double* loss /*TODO Metrics*/);
    void test_step(Batch batch, double* loss /*TODO Metrics*/);
    void learn(int epochs, int batch_size);

private:
    torch::Device device{torch::kCPU};
    std::string data_path, checkpoint_path;
    int max_diffusion_time;
    unet::UNetModel model{nullptr};
    // unet::SimpleUNet model{nullptr};
    std::shared_ptr<torch::optim::Adam> optimizer{nullptr};

    torch::Tensor betas, alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod, posterior_variance, posterior_log_var,
            posterior_mean_coef1, posterior_mean_coef2, sqrt_recip_alphas_cumprod,
            sqrt_recipm1_alphas_cumprod;

    torch::Tensor normal_kl(
            torch::Tensor mean1, torch::Tensor logvar1, torch::Tensor mean2, torch::Tensor logvar2);
};

}  // namespace ddpm

#endif