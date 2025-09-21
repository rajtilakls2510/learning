#ifndef NETWORK_H
#define NETWORK_H

#include <cmath>
#include <tuple>
#include <vector>

#include "network_utils.h"

namespace ddpm {
namespace segformer {

using namespace torch;
using namespace torch::nn::functional;

class OverlapPatchEmbedImpl : public nn::Module {
public:
    OverlapPatchEmbedImpl(int img_size, int patch_size, int stride, int in_chans, int embed_dim);
    std::tuple<Tensor, int, int> forward(Tensor x);

private:
    nn::Conv2d proj{nullptr};
    nn::LayerNorm norm{nullptr};
};

TORCH_MODULE(OverlapPatchEmbed);

class AttentionImpl : public nn::Module {
public:
    AttentionImpl(
            int dim,
            int num_heads,
            bool qkv_bias = false,
            float attn_drop = 0.0,
            float proj_drop = 0.0,
            int sr_ratio = 1);
    Tensor forward(Tensor x, int H, int W);

private:
    int num_heads, dim, sr_ratio;
    float scale;
    nn::Linear q{nullptr}, kv{nullptr}, proj{nullptr};
    nn::Dropout attn_drop_{nullptr}, proj_drop_{nullptr};
    nn::Conv2d sr{nullptr};
    nn::LayerNorm norm{nullptr};
};

TORCH_MODULE(Attention);

class DWConvImpl : public nn::Module {
public:
    DWConvImpl(int dim = 768);
    Tensor forward(Tensor x, int H, int W);

private:
    int dim;
    nn::Conv2d dwconv{nullptr};
};

TORCH_MODULE(DWConv);

class MixFFNImpl : public nn::Module {
public:
    MixFFNImpl(int in_features, int hidden_features, float drop = 0.0);
    Tensor forward(Tensor x, int H, int W);

private:
    nn::Linear fc1{nullptr}, fc2{nullptr};
    DWConv dwconv{nullptr};
    nn::Dropout drop_{nullptr};
};

TORCH_MODULE(MixFFN);

struct DropPathImpl : nn::Module {
    float drop_prob;

    DropPathImpl(float drop_prob_ = 0.0) : drop_prob(drop_prob_) {}
    Tensor forward(const Tensor& x);
};

TORCH_MODULE(DropPath);

class BlockImpl : public nn::Module {
public:
    BlockImpl(
            int dim,
            int num_heads,
            float mlp_ratio = 4.0,
            bool qkv_bias = false,
            float drop = 0.0,
            float attn_drop = 0.0,
            float drop_path = 0.0,
            int sr_ratio = 1);

    Tensor forward(Tensor x, int H, int W);

private:
    nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    DropPath drop_path_{nullptr};
    Attention attn{nullptr};
    MixFFN mlp{nullptr};
};

TORCH_MODULE(Block);

class MixVisionTransformerImpl : public nn::Module {
public:
    MixVisionTransformerImpl(
            int img_size = 224,
            int in_chans = 3,
            std::vector<int> embed_dims = {64, 128, 256, 512},
            std::vector<int> num_heads = {1, 2, 4, 8},
            std::vector<float> mlp_ratios = {4.0, 4.0, 4.0, 4.0},
            bool qkv_bias = false,
            float drop_rate = 0.0,
            float attn_drop_rate = 0.0,
            float drop_path_rate = 0.0,
            std::vector<int> depths = {3, 4, 6, 3},
            std::vector<int> sr_ratios = {8, 4, 2, 1});
    std::vector<Tensor> forward(Tensor x);

private:
    OverlapPatchEmbed patch_embed1{nullptr}, patch_embed2{nullptr}, patch_embed3{nullptr},
            patch_embed4{nullptr};
    nn::LayerNorm norm1{nullptr}, norm2{nullptr}, norm3{nullptr}, norm4{nullptr};
    std::vector<Block> blocks1, blocks2, blocks3, blocks4;
};

TORCH_MODULE(MixVisionTransformer);

class MixVisionTransformerMnistImpl : public nn::Module {
public:
    MixVisionTransformerMnistImpl(
            int img_size = 28,
            int in_chans = 1,
            std::vector<int> embed_dims = {64, 128},
            std::vector<int> num_heads = {2, 4},
            std::vector<float> mlp_ratios = {4.0, 4.0},
            bool qkv_bias = false,
            float drop_rate = 0.0,
            float attn_drop_rate = 0.0,
            float drop_path_rate = 0.0,
            std::vector<int> depths = {3, 3},
            std::vector<int> sr_ratios = {2, 1});
    std::vector<Tensor> forward(Tensor x);

private:
    OverlapPatchEmbed patch_embed1{nullptr}, patch_embed2{nullptr};
    nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    std::vector<Block> blocks1, blocks2;
};

TORCH_MODULE(MixVisionTransformerMnist);

class DecoderMLPImpl : public nn::Module {
public:
    DecoderMLPImpl(int input_dim = 2048, int embed_dim = 768);
    Tensor forward(Tensor x);

private:
    nn::Linear proj{nullptr};
};

TORCH_MODULE(DecoderMLP);

class DecoderImpl : public nn::Module {
public:
    DecoderImpl(
            std::vector<int> in_channels = {64, 128, 256, 512},
            int embed_dim = 256,
            int out_channels = 3,
            float dropout_ratio = 0.1);
    Tensor forward(std::vector<Tensor> x);

private:
    DecoderMLP linear_c4{nullptr}, linear_c3{nullptr}, linear_c2{nullptr}, linear_c1{nullptr};
    nn::Conv2d conv_fuse{nullptr}, conv_pred{nullptr};
    nn::BatchNorm2d bn{nullptr};
    nn::Dropout2d dropout{nullptr};
};

TORCH_MODULE(Decoder);

class DecoderMnistImpl : public nn::Module {
public:
    DecoderMnistImpl(
            std::vector<int> in_channels = {64, 128},
            int embed_dim = 256,
            int out_channels = 1,
            float dropout_ratio = 0.1);
    Tensor forward(std::vector<Tensor> x);

private:
    DecoderMLP linear_c2{nullptr}, linear_c1{nullptr};
    nn::Conv2d conv_fuse{nullptr}, conv_pred{nullptr};
    nn::BatchNorm2d bn{nullptr};
    nn::Dropout2d dropout{nullptr};
};

TORCH_MODULE(DecoderMnist);

class SegFormerImpl : public nn::Module {
public:
    SegFormerImpl(
            int img_size = 224,
            int in_chans = 3,
            std::vector<int> embed_dims = {64, 128, 256, 512},
            std::vector<int> num_heads = {1, 2, 4, 8},
            std::vector<float> mlp_ratios = {4.0, 4.0, 4.0, 4.0},
            bool qkv_bias = false,
            float drop_rate = 0.0,
            float drop_path_rate = 0.1,
            std::vector<int> depths = {3, 4, 6, 3},
            std::vector<int> sr_ratios = {8, 4, 2, 1},
            int decoder_embed_dim = 256,
            int out_channels = 3);
    Tensor forward(Tensor x);

private:
    int img_size;
    MixVisionTransformer encoder{nullptr};
    Decoder decoder{nullptr};
};

TORCH_MODULE(SegFormer);

class SegFormerMnistImpl : public nn::Module {
public:
    SegFormerMnistImpl(
            int img_size = 28,
            int in_chans = 1,
            std::vector<int> embed_dims = {64, 128},
            std::vector<int> num_heads = {2, 4},
            std::vector<float> mlp_ratios = {4.0, 4.0},
            bool qkv_bias = false,
            float drop_rate = 0.0,
            float drop_path_rate = 0.1,
            std::vector<int> depths = {3, 3},
            std::vector<int> sr_ratios = {2, 1},
            int decoder_embed_dim = 256,
            int out_channels = 1);
    Tensor forward(Tensor x);

private:
    int img_size;
    MixVisionTransformerMnist encoder{nullptr};
    DecoderMnist decoder{nullptr};
};

TORCH_MODULE(SegFormerMnist);

}  // namespace segformer

}  // namespace ddpm

#endif