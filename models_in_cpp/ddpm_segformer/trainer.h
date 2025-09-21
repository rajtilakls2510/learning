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

    torch::Tensor q_sample(torch::Tensor x_start, torch::Tensor t, torch::Tensor noise);
    void train_step(Batch batch, double* loss /*TODO Metrics*/);
    void test_step(Batch batch, double* loss /*TODO Metrics*/);
    void learn(int epochs, int batch_size);

private:
    torch::Device device{torch::kCPU};
    std::string data_path, checkpoint_path;
    int max_diffusion_time;
    segformer::SegFormerMnist model{nullptr};
    std::shared_ptr<torch::optim::Adam> optimizer{nullptr};

    torch::Tensor betas, alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod;
};

}  // namespace ddpm

#endif