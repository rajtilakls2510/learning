#ifndef NETWORK_CPP
#define NETWORK_CPP

#include "network.h"

namespace ddpm::unet {

UpsampleImpl::UpsampleImpl(int channels, bool use_conv, int out_channels) : use_conv(use_conv) {
    if (use_conv)
        conv = register_module("conv", Conv2d(Conv2dOptions(channels, out_channels, 3).padding(1)));
}

Tensor UpsampleImpl::forward(Tensor x) {
    x = interpolate(x, InterpolateFuncOptions().scale_factor(std::vector<double>{2.0, 2.0}));
    if (use_conv) x = conv(x);
    return x;
}

DownsampleImpl::DownsampleImpl(int channels, bool use_conv, int out_channels) : use_conv(use_conv) {
    if (use_conv)
        conv = register_module(
                "conv", Conv2d(Conv2dOptions(channels, out_channels, 3).stride(2).padding(1)));
    else
        pool = register_module("pool", AvgPool2d(AvgPool2dOptions({2, 2}).stride({2, 2})));
}

Tensor DownsampleImpl::forward(Tensor x) {
    x = use_conv ? conv(x) : pool(x);
    return x;
}

QKVAttentionImpl::QKVAttentionImpl(int nheads) : nheads(nheads) {}

Tensor QKVAttentionImpl::forward(Tensor qkv) {
    // qkv: [B x (3 * nheads * C) x T] tensor of Qs, Ks, and Vs.
    int bs = qkv.size(0);
    int width = qkv.size(1);
    int length = qkv.size(2);

    int ch = width / (3 * nheads);
    auto q_k_v_ = qkv.chunk(3, /*dim*/ 1);
    Tensor q = q_k_v_[0];
    Tensor k = q_k_v_[1];
    Tensor v = q_k_v_[2];

    float scale = (float)(1.0 / std::sqrt(ch));  // Maybe make two sqrt
    Tensor weight = torch::einsum("bct,bcs->bts", {q * scale, k * scale});
    weight = weight.softmax(/*dim*/ -1);
    Tensor a = torch::einsum("bts,bcs->bct", {weight, v});
    return a.reshape({bs, -1, length});  // [B x (nheads * C) x T]
}

AttentionBlockImpl::AttentionBlockImpl(int channels, int nheads) {
    norm = register_module("norm", GroupNorm(GroupNormOptions(32, channels)));
    qkv = register_module("qkv", Conv1d(Conv1dOptions(channels, 3 * channels, 1)));
    attention = register_module("attention", QKVAttention(nheads));
    proj_out = register_module("proj_out", Conv1d(Conv1dOptions(channels, channels, 1)));
    
    // Zero init
    init::zeros_(proj_out->weight);
    if (proj_out->options.bias()) {
        init::zeros_(proj_out->bias);
    }
}

Tensor AttentionBlockImpl::forward(Tensor x) {
    int b = x.size(0);
    int c = x.size(1);
    Tensor x_ = x.reshape({b, c, -1});
    Tensor qkv_ = qkv(norm(x_));
    Tensor h = proj_out(attention(qkv_));
    return (x_+h).reshape(x.sizes());
}

}  // namespace ddpm::unet
#endif