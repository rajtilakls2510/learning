#include <iostream>
#include "network.h"

using namespace ddpm;

int main(int argc, char* argv[]) {

    torch::Device device(torch::kCUDA);

    // // Time Encoding Test 
    // auto time_enc = TimePosEncoding(128);
    // torch::Tensor t = torch::randint(1, 1001, {16}, torch::kLong).to(device);
    // auto enc = time_enc->forward(t);
    // std::cout << "enc : " << net::get_size(enc) << "\n";
    
    // // ViTBlock Test
    // torch::Tensor t = torch::randint(1, 1001, {16}, torch::kLong).to(device);
    // torch::Tensor x = torch::zeros({12, 16, 128}).to(device);
    // auto vit_block = ViTBlock(128, 4, 512, 128);
    // vit_block->to(device);
    // auto vb = vit_block->forward(x, t);
    // std::cout << "vb: " << net::get_size(vb) << "\n";
     
    // ViT Test
    torch::Tensor t = torch::randint(1, 1001, {16}, torch::kLong).to(device);
    torch::Tensor x = torch::zeros({16, 1, 28, 28}).to(device);
    auto vit = ViT(/*img_size*/ 28, /*patch_size*/ 4);
    vit->to(device);
    auto v = vit->forward(x, t);
    std::cout << "v: " << net::get_size(v) << "\n";
    


    return 0;
}

