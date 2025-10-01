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
    return torch::cat({args.sin(), args.cos()}, /*dim*/ -1);  // [B, dim]
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

UNetUpsampleImpl::UNetUpsampleImpl(int in_ch, int out_ch, int time_dim, int kernel_size)
    : in_ch(in_ch), out_ch(out_ch), time_dim(time_dim), kernel_size(kernel_size) {
    time_encoder = register_module("time_encoder", TimePosEncoding(time_dim));
    conv_in = register_module(
            "conv_in", Conv2d(Conv2dOptions(2 * in_ch, out_ch, kernel_size).padding(1).bias(true)));
    spatial = register_module(
            "spatial",
            ConvTranspose2d(
                    ConvTranspose2dOptions(out_ch, out_ch, 4).stride(2).padding(1).bias(true)));
    bn1 = register_module("bn1", BatchNorm2d(BatchNorm2dOptions(out_ch)));
    bn2 = register_module("bn2", BatchNorm2d(BatchNorm2dOptions(out_ch)));
    conv_feat = register_module(
            "conv_feat", Conv2d(Conv2dOptions(out_ch, out_ch, 3).padding(1).bias(true)));
    time_proj = register_module("time_proj", Linear(time_dim, out_ch));
}

torch::Tensor UNetUpsampleImpl::forward(torch::Tensor x, torch::Tensor t) {
    // x : [B, C, H, W] t : [B,]
    x = bn1(functional::relu(conv_in(x)));
    torch::Tensor t_emb = relu(time_proj(time_encoder(t)));
    x += t_emb.unsqueeze(-1).unsqueeze(-1);
    x = bn2(functional::relu(conv_feat(x)));
    return spatial(x);
}

UNetImpl::UNetImpl(
        int img_size, int img_channels, int time_dim, std::vector<int> channel_sequence) {
    stem = register_module(
            "stem",
            Conv2d(Conv2dOptions(img_channels, channel_sequence[0], 3).padding(1).bias(true)));

    for (int i = 1; i < channel_sequence.size(); i++) {
        auto down_block = UNetDownsample(
                channel_sequence[i - 1], channel_sequence[i], time_dim, /*kernel_size*/ 3);
        down_blocks.push_back(down_block);
        register_module("down_block_" + std::to_string(i), down_block);
    }

    for (int i = (int)channel_sequence.size() - 1; i > 0; i--) {
        auto up_block = UNetUpsample(
                channel_sequence[i], channel_sequence[i - 1], time_dim, /*kernel_size*/ 3);
        up_blocks.push_back(up_block);
        register_module("up_block_" + std::to_string(i), up_block);
    }

    head = register_module(
            "head", Conv2d(Conv2dOptions(channel_sequence[0], img_channels, 1).bias(true)));
}

torch::Tensor UNetImpl::forward(torch::Tensor x, torch::Tensor t) {
    // x : [B, img_channels, H, W], t : [B,]

    std::vector<torch::Tensor> skips;
    torch::Tensor h = stem(x);

    for (int i = 0; i < down_blocks.size(); i++) {
        h = down_blocks[i](h, t);
        skips.push_back(h);
    }

    for (int i = 0; i < up_blocks.size(); i++) {
        h = up_blocks[i](torch::cat({h, skips[(int)up_blocks.size() - i - 1]}, /*dim*/ 1), t);
    }
    return head(h);
}

}  // namespace ddpm

#endif