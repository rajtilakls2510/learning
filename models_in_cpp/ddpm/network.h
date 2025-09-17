#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

#include "network_utils.h"

namespace ddpm {

using namespace torch::nn;
using namespace net;
using namespace torch::indexing;

torch::Tensor posemb_sincos_2d(int h, int w, int dim, int temp = 10000);
torch::Tensor patchify(torch::Tensor x, int p1, int p2);
torch::Tensor depatchify(torch::Tensor patches, int p, int H, int W, int C);

class TimePosEncodingImpl : public Module {
public:
    TimePosEncodingImpl(int dim);
    torch::Tensor forward(torch::Tensor t);

private:
    int dim;
};

TORCH_MODULE(TimePosEncoding);

class ViTBlockImpl : public Module {
public:
    ViTBlockImpl(int dim, int heads, int mlp_dim, int time_dim);
    torch::Tensor forward(torch::Tensor x, torch::Tensor t);

private:
    int dim, heads, mlp_dim, time_dim;
    TimePosEncoding time_encoder{nullptr};
    TransformerEncoderLayer encoder{nullptr};
    Linear time_proj{nullptr};
};

TORCH_MODULE(ViTBlock);

class ViTImpl : public torch::nn::Module {
public:
    ViTImpl(int img_size = 224,
            int patch_size = 16,
            int dim = 128,
            int depth = 4,
            int heads = 4,
            int mlp_dim = 512,
            int time_dim = 128,
            int channels = 1);

    torch::Tensor forward(torch::Tensor x, torch::Tensor t);

private:
    int patch_size, img_size, channels, depth;
    torch::nn::Sequential to_patch_embedding{nullptr};
    torch::Tensor pos_embedding{nullptr};
    torch::nn::Sequential reconstruction_head{nullptr};
    std::vector<ViTBlock> blocks;
};

TORCH_MODULE(ViT);
}  // namespace ddpm

#endif