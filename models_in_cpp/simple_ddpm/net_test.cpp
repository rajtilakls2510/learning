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
    
    // // UNetDownsample Test
    // torch::Tensor t = torch::randint(1, 1001, {16}, torch::kLong).to(device);
    // torch::Tensor x = torch::zeros({16, 2, 32, 32}).to(device);
    // auto unet_down = UNetDownsample(/*in_ch*/2, /*out_ch*/4, /*time_dim*/8, /*kernel_size*/3);
    // unet_down->to(device);
    // auto u = unet_down->forward(x, t);
    // std::cout << "u: " << net::get_size(u) << "\n";
    
    // // UNetUpsample Test
    // torch::Tensor t = torch::randint(1, 1001, {16}, torch::kLong).to(device);
    // torch::Tensor x = torch::zeros({16, 8, 32, 32}).to(device);
    // auto unet_up = UNetUpsample(/*in_ch*/4, /*out_ch*/2, /*time_dim*/8, /*kernel_size*/3);
    // unet_up->to(device);
    // auto u = unet_up->forward(x, t);
    // std::cout << "u: " << net::get_size(u) << "\n";
    

    // UNet Test
    torch::Tensor t = torch::randint(1, 1001, {16}, torch::kLong).to(device);
    torch::Tensor x = torch::zeros({16, 1, 28, 28}).to(device);
    auto unet = UNet(/*img_size*/28, /*img_channels*/1, /*time_dim*/8);
    unet->to(device);
    auto u = unet->forward(x, t);
    std::cout << "u: " << net::get_size(u) << "\n";
    


    return 0;
}

