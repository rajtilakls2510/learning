#ifndef TRAINER_H
#define TRAINER_H

#include <filesystem>
#include <string>

#include "mnist_loader.hpp"
#include "network.h"

namespace vdm {
using namespace MNIST;

class Trainer {
public:
    Trainer(std::string data_path,
            std::string checkpoint_path,
            int max_diffusion_time,
            bool use_cpu = false);

    torch::Tensor encode(torch::Tensor x);
    torch::Tensor decode(torch::Tensor z, torch::Tensor g_0);
    torch::Tensor logprob(torch::Tensor x, torch::Tensor z, torch::Tensor g_0);
    void train_step(Batch batch, double* model_loss, double* classifier_loss);
    void test_step(Batch batch, double* model_loss, double* classifier_loss);
    void learn(int epochs, int batch_size);

private:
    double gamma_min{-13.3}, gamma_max{5.0};
    int vocab_size{255};
    torch::Device device{torch::kCPU};
    std::string data_path, checkpoint_path;
    int max_diffusion_time;
    unet::ScoreModel model{nullptr};
    unet::NoiseNet gamma{nullptr};
    std::shared_ptr<torch::optim::Adam> optimizer{nullptr}, gamma_optimizer{nullptr};
};

}  // namespace vdm

#endif