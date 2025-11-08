#ifndef NETWORK_H
#define NETWORK_H

#include <cmath>
#include <vector>

#include "network_utils.h"

namespace vdm::unet {

using namespace net;
using namespace torch;
using namespace torch::nn;
using namespace torch::indexing;
using namespace torch::nn::functional;

Tensor timestep_embedding(Tensor t, int dim, int max_diffusion_time);

class DenseMonotoneImpl : public Module {
public:
    DenseMonotoneImpl(int in_features, int out_features, bool use_bias = true);
    Tensor forward(Tensor x);

    Tensor weight, bias;

private:
    bool use_bias;
};
TORCH_MODULE(DenseMonotone);

class NoiseNetImpl : public Module {
public:
    NoiseNetImpl(int mid_features = 1024, double gamma_min = 0.0, double gamma_max = 10.0);
    Tensor forward(Tensor x);

public:
    int mid_features;
    double gamma_min, gamma_max;
    DenseMonotone l1{nullptr}, l2{nullptr}, l3{nullptr};
};
TORCH_MODULE(NoiseNet);

class ResnetBlockImpl : public Module {
public:
    ResnetBlockImpl(int in_channels, int out_channels, int cond_dim);
    Tensor forward(Tensor x, Tensor cond);

private:
    int in_channels, out_channels;
    Conv2d conv1{nullptr}, conv2{nullptr}, min_shortcut{nullptr};
    Linear cond_proj{nullptr};
    Dropout drop{nullptr};
    GroupNorm gpn1{nullptr}, gpn2{nullptr};
};
TORCH_MODULE(ResnetBlock);

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

class ScoreModelImpl : public Module {
public:
    ScoreModelImpl(int in_out_channels, int n_res_layers, int n_embed, int max_diffusion_time);
    Tensor forward(Tensor z, Tensor g_t, Tensor gamma_min, Tensor gamma_max);

private:
    int n_res_layers, n_embed, max_diffusion_time;

    Linear dense0{nullptr}, dense1{nullptr};
    Conv2d conv_in{nullptr}, conv_out{nullptr};
    std::vector<ResnetBlock> down_blocks;
    std::vector<ResnetBlock> up_blocks;
    ResnetBlock midres1_block{nullptr}, midres2_block{nullptr};
    AttentionBlock mid_attn_block{nullptr};
    GroupNorm norm{nullptr};
};
TORCH_MODULE(ScoreModel);

}  // namespace vdm::unet

#endif