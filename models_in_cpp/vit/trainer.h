#ifndef TRAINER_H
#define TRAINER_H

#include "vit.h"
#include "data_loader.h"
#include <filesystem>
#include <memory>

namespace trainer { 
    using namespace MNIST;

    class Trainer {
    public:
        Trainer(std::string data_path, std::string checkpoint_path);
        void learn(int epochs, int batch_size);
        void train_step(Batch batch, double* loss, double* accuracy);
        void test_step(Batch batch, double* loss, double* accuracy);
    private:
        std::string data_path, checkpoint_path;
        vit::ViT model{nullptr};
        std::shared_ptr<torch::optim::Adam> optimizer{nullptr};
    };
}

#endif