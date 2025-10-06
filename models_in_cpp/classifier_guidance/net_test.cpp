#include <iostream>

#include "network.h"

using namespace ddpm::unet;

int main(int argc, char* argv[]) {
    NoGradGuard no_grad;

    // // Upsample test
    // auto up = ddpm::unet::Upsample(4, true, 6);
    // Tensor inp = torch::zeros({8, 4, 7,7});
    // Tensor logits = up(inp);
    // std::cout << get_size(logits) << "\n";

    // // Downsample test
    // auto down = ddpm::unet::Downsample(6, false, 4);
    // Tensor inp = torch::zeros({8,6,14,14});
    // Tensor logits = down(inp);
    // std::cout << get_size(logits) << "\n";

    // // QKVAttention test
    // auto qkv_attn = QKVAttention(4);
    // Tensor inp = torch::zeros({8, (3*4*4), 64});
    // Tensor logits = qkv_attn(inp);
    // std::cout << get_size(logits) << "\n";

    // // AttentionBlock test
    // auto attn = AttentionBlock(4*2*32, 4);
    // Tensor inp = torch::zeros({8, 4*2*32, 64});
    // Tensor logits = attn(inp);
    // std::cout << get_size(logits) << "\n";

    // // ResBlock test
    // auto resblock = ResBlock(32, 128, 0.1, 64, true, false, true);
    // Tensor inp = torch::zeros({8, 32, 16, 16});
    // Tensor inp2 = torch::zeros({8, 128});
    // Tensor logits = resblock(inp, inp2);
    // std::cout << get_size(logits) << "\n";

    // UNetModel test
    // auto unet = UNetModel(224, 3, 64, 1, 2, 0.1, 4);
    // Tensor inp = torch::zeros({8, 3, 224, 224});
    // Tensor t = torch::randint(1, 1001, {8}, torch::kLong);
    // Tensor logits = unet(inp, t);
    // std::cout << get_size(logits) << "\n";

    // std::cout << "Num parameters: " << count_parameters(unet) << "\n";

    // // UNetModel test for MNIST
    // std::vector<int> cm = {2, 4, 8};
    // auto unet = UNetModel(28, 1, 32, 1, 2, 0.1, 4, 1, cm);
    // Tensor inp = torch::zeros({8, 1, 28, 28});
    // Tensor t = torch::randint(1, 1001, {8}, torch::kLong);
    // Tensor logits = unet(inp, t);
    // std::cout << get_size(logits) << "\n";

    // std::cout << "Num parameters: " << count_parameters(unet) << "\n";

    // SimpleUNetClassifier test for MNIST
    torch::Tensor t = torch::randint(1, 1001, {16}, torch::kLong);
    torch::Tensor x = torch::zeros({16, 1, 28, 28});
    auto unet = SimpleUNetClassifier(
            /*img_size*/ 28, /*img_channels*/ 1, /*num_classes*/ 10, /*time_dim*/ 8);
    auto u = unet->forward(x, t);
    std::cout << "u: " << get_size(u) << "\n";
    std::cout << "Num parameters: " << count_parameters(unet) << "\n";
    return 0;
}