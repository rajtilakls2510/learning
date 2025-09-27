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

    // AttentionBlock test
    auto attn = AttentionBlock(4*2*32, 4);
    Tensor inp = torch::zeros({8, 4*2*32, 64});
    Tensor logits = attn(inp);
    std::cout << get_size(logits) << "\n"; 
    return 0;
}