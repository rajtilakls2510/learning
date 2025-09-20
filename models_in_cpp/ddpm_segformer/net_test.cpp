#include <iostream>

#include "network.h"

using namespace ddpm;

int main(int argc, char* argv[]) {
    torch::NoGradGuard no_grad;

    // // OverlapPatchEmbed test
    // auto ope = OverlapPatchEmbed(
    //         /*img_size*/ 224, /*patch_size*/ 7, /*stride*/ 4, /*in_chans*/ 3, /*embed_dim*/ 768);
    // torch::Tensor x = torch::zeros({16, 3, 224, 224});
    // std::tuple<torch::Tensor, int, int> out = ope(x);
    // Tensor logit = std::get<0>(out);
    // int H = std::get<1>(out);
    // int W = std::get<2>(out);
    // std::cout << net::get_size(logit) << " H: " << H << " W: " << W << "\n";

    // // Attention test
    // auto att = Attention(
    //         /*dim*/ 32,
    //         /*num_heads*/ 8,
    //         /*qkv_bias*/ false,
    //         /*attn_drop*/ 0.0,
    //         /*proj_drop*/ 0.0,
    //         /*sr_ratio*/ 2);
    // torch::Tensor x = torch::zeros({16, 16 * 16, 32});
    // auto logit = att(x, 16, 16);
    // std::cout << net::get_size(logit) << "\n";

    // // DWConv test
    // auto dwconv = DWConv(/*dim*/ 32);
    // torch::Tensor x = torch::zeros({16, 16 * 16, 32});
    // auto logit = dwconv(x, 16, 16);
    // std::cout << net::get_size(logit) << "\n";

    // // MixFFN test
    // auto mffn = MixFFN(/*in_features*/ 32, /*hidden_features*/ 64);
    // torch::Tensor x = torch::zeros({16, 16 * 16, 32});
    // auto logit = mffn(x, 16, 16);
    // std::cout << net::get_size(logit) << "\n";

    // // DropPath test
    // auto dropp = DropPath(/*drop_prob*/ 0.1);
    // dropp->train();
    // torch::Tensor x = torch::zeros({16, 16 * 16, 32});
    // auto logit = dropp(x);
    // std::cout << net::get_size(logit) << "\n";

    // // Block test
    // auto block = Block(/*dim*/ 64, /*num_heads*/ 2);
    // torch::Tensor x = torch::zeros({16, 16 * 16, 64});
    // auto logit = block(x, 16, 16);
    // std::cout << net::get_size(logit) << "\n";

    // // MixVisionTransformer test
    // auto mvt = MixVisionTransformer();
    // torch::Tensor x = torch::zeros({16, 3, 224, 224});
    // auto logits = mvt(x);
    // std::cout << net::get_size(logits[0]) << "\n";
    // std::cout << net::get_size(logits[1]) << "\n";
    // std::cout << net::get_size(logits[2]) << "\n";
    // std::cout << net::get_size(logits[3]) << "\n";

    // MixVisionTransformer test for MNIST
    auto mvt = MixVisionTransformerMnist(28, 1);
    torch::Tensor x = torch::zeros({16, 1, 28, 28});
    auto logits = mvt(x);
    std::cout << net::get_size(logits[0]) << "\n";
    std::cout << net::get_size(logits[1]) << "\n";

}