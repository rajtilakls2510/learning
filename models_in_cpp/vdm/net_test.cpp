#include <iostream>

#include "network.h"

using namespace vdm::unet;

int main(int argc, char* argv[]) {
    NoGradGuard no_grad;

    // // DenseMonotone test
    // auto dm = DenseMonotone(64, 32);
    // Tensor inp = torch::full({8, 64}, 2.0);
    // Tensor logits = dm(inp);
    // std::cout << get_size(logits) << "\n";
    // std::cout << logits << "\n";

    // // NoiseNet test
    // auto nnet = NoiseNet();
    // Tensor inp = torch::full({8, 1}, 0.5);
    // Tensor logits = nnet(inp);
    // std::cout << get_size(logits) << "\n";
    // std::cout << logits << "\n";

    // // ResnetBlock test
    // auto resblock = ResnetBlock(32, 32, 16);
    // Tensor inp = torch::zeros({8, 32, 8, 8});
    // Tensor cond = torch::zeros({8, 16});
    // Tensor logits = resblock(inp, cond);
    // std::cout << get_size(logits) << "\n";

    // // AttentionBlock test
    // auto attn = AttentionBlock(32, 2);
    // Tensor inp = torch::zeros({8, 32, 64, 64});
    // Tensor logits = attn(inp);
    // std::cout << get_size(logits) << "\n";

    // ScoreModel test
    auto score = ScoreModel(1, 3, 256, 1000);
    Tensor inp = torch::zeros({8, 1, 28, 28});
    Tensor g_t = torch::tensor({-13.0, -11.0, -8.0, -5.0, -2.0, 0.0, 1.0, 3.0});
    Tensor logits = score(inp, g_t, g_t, g_t);
    std::cout << get_size(logits) << "\n";
    std::cout << "Num parameters: " << count_parameters(score) << "\n";

    return 0;
}