#include <iostream>

#include "network.h"

using namespace ddpm::unet;

int main(int argc, char* argv[]) {
    NoGradGuard no_grad;

    // SimpleUNet test for MNIST
    int num_classes = 10;
    int img_size = 28;
    int bs = 16;
    torch::Tensor t = torch::randint(1, 1001, {bs}, torch::kLong);
    torch::Tensor x = torch::zeros({bs, 1, img_size, img_size});
    torch::Tensor class_ids = torch::full({bs}, 3, torch::kLong);
    auto unet = SimpleUNet(
            img_size, /*in_channels*/ 1, /*out_channels*/ 1, /*time_dim*/ 8, num_classes);
    auto u = unet(x, t, class_ids);
    std::cout << "u: " << get_size(u) << "\n";
    std::cout << "Num parameters: " << count_parameters(unet) << "\n";

    return 0;
}