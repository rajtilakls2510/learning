#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

#include "network_utils.h"

namespace ddpm {

using namespace net;
using namespace torch::indexing;

torch::Tensor posemb_sincos_2d(int h, int w, int dim, int temp = 10000);
torch::Tensor patchify(torch::Tensor x, int p1, int p2);
torch::Tensor depatchify(torch::Tensor patches, int p, int H, int W, int C);

class ViTImpl : public torch::nn::Module {
public:
    ViTImpl(int img_size = 224,
            int patch_size = 16,
            int dim = 128,
            int depth = 4,
            int heads = 4,
            int mlp_dim = 512,
            int channels = 3,
            int max_diffusion_time = 1000);

    torch::Tensor forward(torch::Tensor x, torch::Tensor t);

private:
    int patch_size, img_size, channels;
    torch::nn::Sequential to_patch_embedding{nullptr};
    torch::nn::Embedding embedding_time{nullptr};
    torch::nn::Sequential time_mlp{nullptr};
    torch::Tensor pos_embedding{nullptr};
    torch::Tensor time_pos_embedding{nullptr};
    torch::nn::TransformerEncoder image_encoder{nullptr};
    torch::nn::Sequential reconstruction_head{nullptr};
};

TORCH_MODULE(ViT);
}  // namespace ddpm

#endif