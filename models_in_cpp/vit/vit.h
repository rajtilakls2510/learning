#ifndef VIT_H
#define VIT_H

#include <vector>

#include "network_utils.h"

namespace vit {
using namespace net;
using namespace torch::indexing;

torch::Tensor posemb_sincos_2d(int h, int w, int dim, int temp = 10000);
torch::Tensor patchify(torch::Tensor x, int p1, int p2);

class ViTImpl : public torch::nn::Module {
public:
    torch::nn::Sequential to_patch_embedding{nullptr};
    torch::Tensor pos_embedding{nullptr};
    torch::nn::TransformerEncoder encoder{nullptr};
    torch::nn::Sequential class_head{nullptr};
    int patch_size;

    ViTImpl(int img_size = 224,
            int patch_size = 16,
            int num_classes = 10,
            int dim = 128,
            int depth = 4,
            int heads = 4,
            int mlp_dim = 512,
            int channels = 3);

    torch::Tensor forward(torch::Tensor x);
};

TORCH_MODULE(ViT);

}  // namespace vit

#endif