#ifndef NETWORK_H
#define NETWORK_H

#include <cmath>
#include <vector>

#include "network_utils.h"

namespace ddpm::unet {

using namespace net;
using namespace torch;
using namespace torch::nn;
using namespace torch::indexing;
using namespace torch::nn::functional;

class UpsampleImpl : public Module {
public:
    UpsampleImpl(int channels, bool use_conv, int out_channels = 1);
    Tensor forward(Tensor x);

private:
    bool use_conv;
    Conv2d conv{nullptr};
};
TORCH_MODULE(Upsample);

class DownsampleImpl : public Module {
public:
    DownsampleImpl(int channels, bool use_conv, int out_channels = 1);
    Tensor forward(Tensor x);

private:
    bool use_conv;
    Conv2d conv{nullptr};
    AvgPool2d pool{nullptr};
};
TORCH_MODULE(Downsample);

class QKVAttentionImpl : public Module {
public:
    QKVAttentionImpl(int nheads);
    Tensor forward(Tensor qkv);

private:
    int nheads;
};
TORCH_MODULE(QKVAttention);

class AttentionBlockImpl : public Module {
public:
    AttentionBlockImpl(int channels, int nheads);
    Tensor forward(Tensor x);

private:
    GroupNorm norm{nullptr};
    Conv1d qkv{nullptr}, proj_out{nullptr};
    QKVAttention attention{nullptr};
};
TORCH_MODULE(AttentionBlock);

}  // namespace ddpm::unet

#endif