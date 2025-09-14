#ifndef TRAINER_H
#define TRAINER_H

#include <filesystem>
#include <memory>

#include "mnist_loader.hpp"
#include "vit.h"

namespace trainer {
using namespace MNIST;

class Trainer {
public:
    Trainer(std::string data_path, std::string checkpoint_path, bool use_cpu = false);
    void learn(int epochs, int batch_size);
    void train_step(Batch batch, double* loss, double* accuracy);
    void test_step(Batch batch, double* loss, double* accuracy);

private:
    torch::Device device{torch::kCPU};
    std::string data_path, checkpoint_path;
    vit::ViT model{nullptr};
    std::shared_ptr<torch::optim::Adam> optimizer{nullptr};
};
}  // namespace trainer

#endif