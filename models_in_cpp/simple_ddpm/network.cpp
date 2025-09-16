#ifndef NETWORK_CPP
#define NETWORK_CPP

#include "network.h"

namespace ddpm {

TimePosEncodingImpl::TimePosEncodingImpl(int dim) : dim(dim) {}

torch::Tensor TimePosEncodingImpl::forward(torch::Tensor t) {
    // t : [B,]
    int half_dim = dim / 2;
    torch::Tensor scale = torch::log(torch::tensor((float)10000)) / (half_dim - 1);
    torch::Tensor exponents = torch::exp(torch::arange(half_dim).to(t.device()) * -scale);
    torch::Tensor args = t.unsqueeze(-1) * exponents.unsqueeze(0);
    return torch::cat({args.sin(), args.cos()}, /*dim*/ -1);    // [B, dim]
}

UNetDownsampleImpl::UNetDownsampleImpl(int in_ch, int out_ch, int time_dim, int kernel_size)
    : in_ch(in_ch), out_ch(out_ch), time_dim(time_dim), kernel_size(kernel_size) {
    time_encoder = register_module("time_encoder", TimePosEncoding(time_dim));
    conv_in = register_module(
            "conv_in", Conv2d(Conv2dOptions(in_ch, out_ch, kernel_size).padding(1).bias(true)));
    spatial = register_module(
            "spatial", Conv2d(Conv2dOptions(out_ch, out_ch, 4).stride(2).padding(1).bias(true)));
    bn1 = register_module("bn1", BatchNorm2d(BatchNorm2dOptions(out_ch)));
    bn2 = register_module("bn2", BatchNorm2d(BatchNorm2dOptions(out_ch)));
    conv_feat = register_module(
            "conv_feat", Conv2d(Conv2dOptions(out_ch, out_ch, 3).padding(1).bias(true)));
    time_proj = register_module("time_proj", Linear(time_dim, out_ch));
}

torch::Tensor UNetDownsampleImpl::forward(torch::Tensor x, torch::Tensor t) {
    // x : [B, C, H, W] t : [B,]
    x = bn1(functional::relu(conv_in(x)));
    torch::Tensor t_emb = relu(time_proj(time_encoder(t)));
    x += t_emb.unsqueeze(-1).unsqueeze(-1);
    x = bn2(functional::relu(conv_feat(x)));
    return spatial(x);
}

}  // namespace ddpm

#endif