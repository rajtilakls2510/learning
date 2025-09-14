#include <iostream>
#include "vit.h"
#include "mnist_loader.hpp"
#include "trainer.h"

using namespace trainer;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./train <data_path> <checkpoint_path>\n";
        return -1;
    }

    Trainer trainer(argv[1], argv[2]);
    trainer.learn(10, 16);
}