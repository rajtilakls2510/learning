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

Tensor timestep_embedding(Tensor t, int dim);

class UpsampleImpl : public Module {
public:
    UpsampleImpl(int channels, bool use_conv, int out_channels = 1);
    Tensor forward(Tensor x);

private:
    bool use_conv;
    Conv2d conv{nullptr};
    BatchNorm2d bn{nullptr};
    ConvTranspose2d spatial{nullptr};
};
TORCH_MODULE(Upsample);

class DownsampleImpl : public Module {
public:
    DownsampleImpl(int channels, bool use_conv, int out_channels = 1);
    Tensor forward(Tensor x);

private:
    bool use_conv;
    Conv2d conv{nullptr};
    BatchNorm2d bn{nullptr};
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

class ResBlockImpl : public Module {
public:
    ResBlockImpl(
            int channels,
            int emb_channels,
            float drop,
            int out_channels,
            bool use_conv = false,
            bool up = false,
            bool down = false);
    Tensor forward(Tensor x, Tensor emb);

private:
    int channels, out_channels;
    bool up, down, up_down;
    Sequential in_layers{nullptr}, emb_layers{nullptr}, out_layers{nullptr};
    Upsample h_up{nullptr}, x_up{nullptr};
    Downsample h_down{nullptr}, x_down{nullptr};
    Conv2d skip_connection{nullptr}, in_conv{nullptr};
    GroupNorm out_norm{nullptr};
    BatchNorm2d skip_bn{nullptr};
};
TORCH_MODULE(ResBlock);

class UNetBlockDownImpl : public Module {
public:
    UNetBlockDownImpl(
            int in_channels,
            int time_embed_dim,
            int num_heads,
            int num_res_blocks,
            int out_channels,
            float dropout,
            bool use_attn = true,
            bool use_down = true);
    Tensor forward(Tensor x, Tensor t);

private:
    int num_res_blocks;
    bool use_attn, use_down;
    std::vector<ResBlock> res_no_downs;
    ResBlock res_down{nullptr};
    std::vector<AttentionBlock> attns;
};
TORCH_MODULE(UNetBlockDown);

class UNetBlockUpImpl : public Module {
public:
    UNetBlockUpImpl(
            int in_channels,
            int time_embed_dim,
            int num_heads,
            int num_res_blocks,
            int out_channels,
            float dropout,
            bool use_attn = false,
            bool use_up = false);
    Tensor forward(Tensor x, Tensor t);

private:
    int num_res_blocks;
    bool use_attn, use_up;
    std::vector<ResBlock> res_no_ups;
    ResBlock res_up{nullptr};
    std::vector<AttentionBlock> attns;
};
TORCH_MODULE(UNetBlockUp);

class UNetModelImpl : public Module {
public:
    UNetModelImpl(
            int img_size,
            int in_channels,
            int model_channels,
            int out_channels,
            int num_res_blocks,
            float dropout,
            int num_heads,
            int begin_attn = 1,
            std::vector<int> channel_mult = {1, 2, 4, 8});
    Tensor forward(Tensor x, Tensor t);

private:
    int model_channels, num_res_blocks;
    std::vector<int> channel_mult;
    Sequential time_embed{nullptr}, out{nullptr};
    Conv2d inp{nullptr};
    std::vector<UNetBlockDown> down_blocks;
    std::vector<UNetBlockUp> up_blocks;

    AttentionBlock mid_attn{nullptr};
    ResBlock mid_res1{nullptr}, mid_res2{nullptr};
};
TORCH_MODULE(UNetModel);

class TimePosEncodingImpl : public Module {
public:
    TimePosEncodingImpl(int dim);
    torch::Tensor forward(torch::Tensor t);

private:
    int dim;
};

TORCH_MODULE(TimePosEncoding);

class UNetDownsampleImpl : public Module {
public:
    UNetDownsampleImpl(int in_ch, int out_ch, int time_dim, int kernel_size);
    torch::Tensor forward(torch::Tensor x, torch::Tensor t);

private:
    int in_ch, out_ch, time_dim, kernel_size;
    torch::Tensor time_pos_embedding{nullptr};
    Conv2d conv_in{nullptr}, spatial{nullptr}, conv_feat{nullptr};
    BatchNorm2d bn1{nullptr}, bn2{nullptr};
    Linear time_proj{nullptr};
    TimePosEncoding time_encoder{nullptr};
};

TORCH_MODULE(UNetDownsample);

class UNetUpsampleImpl : public Module {
public:
    UNetUpsampleImpl(int in_ch, int out_ch, int time_dim, int kernel_size);
    torch::Tensor forward(torch::Tensor x, torch::Tensor t);

private:
    int in_ch, out_ch, time_dim, kernel_size;
    torch::Tensor time_pos_embedding{nullptr};
    Conv2d conv_in{nullptr}, conv_feat{nullptr};
    ConvTranspose2d spatial{nullptr};
    BatchNorm2d bn1{nullptr}, bn2{nullptr};
    Linear time_proj{nullptr};
    TimePosEncoding time_encoder{nullptr};
};

TORCH_MODULE(UNetUpsample);

class SimpleUNetImpl : public torch::nn::Module {
public:
    SimpleUNetImpl(
            int img_size,
            int in_channels,
            int out_channels,
            int time_dim,
            std::vector<int> channel_sequence = {32, 64, 128});
    torch::Tensor forward(torch::Tensor x, torch::Tensor t);

private:
    std::vector<int> channel_sequence;
    Conv2d stem{nullptr}, head{nullptr};
    std::vector<UNetDownsample> down_blocks;
    std::vector<UNetUpsample> up_blocks;
};

TORCH_MODULE(SimpleUNet);

}  // namespace ddpm::unet

#endif