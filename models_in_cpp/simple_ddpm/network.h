#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

#include "network_utils.h"

namespace ddpm {

using namespace net;
using namespace torch::nn;
using namespace torch::indexing;

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

class UNetImpl : public torch::nn::Module {
public:
    UNetImpl(
            int img_size,
            int img_channels,
            int time_dim,
            std::vector<int> channel_sequence = {64, 256, 512});
    torch::Tensor forward(torch::Tensor x, torch::Tensor t);
private:
    int img_size, img_channels, time_dim;
    std::vector<int> channel_sequence;
    Conv2d stem{nullptr}, head{nullptr};
    std::vector<UNetDownsample> down_blocks;
    std::vector<UNetUpsample> up_blocks;
};

TORCH_MODULE(UNet);

}  // namespace ddpm

#endif