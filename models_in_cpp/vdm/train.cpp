#include <iostream>

#include "trainer.h"

using namespace vdm;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./train <data_path> <checkpoint_path>\n";
        return -1;
    }

    Trainer trainer(argv[1], argv[2], /* max diffusion time */ 1000);
    trainer.learn(/* epochs */ 100, /* batch_size*/ 32);
}