#ifndef NETWORK_CPP
#define NETWORK_CPP

#include "network.h"

namespace ddpm {

OverlapPatchEmbedImpl::OverlapPatchEmbedImpl(
        int img_size, int patch_size, int stride, int in_chans, int embed_dim) {
    int H = img_size / patch_size;
    int W = H;

    proj = register_module(
            "proj",
            nn::Conv2d(nn::Conv2dOptions(in_chans, embed_dim, patch_size)
                               .stride(stride)
                               .padding(patch_size / 2)));
    norm = register_module("norm", nn::LayerNorm(nn::LayerNormOptions({embed_dim})));
}

std::tuple<Tensor, int, int> OverlapPatchEmbedImpl::forward(Tensor x) {
    x = proj(x);
    int H = x.size(2);
    int W = x.size(3);
    x = x.flatten(2);
    x = x.transpose(1, 2);
    x = norm(x);
    return {x, H, W};
}

AttentionImpl::AttentionImpl(
        int dim, int num_heads, bool qkv_bias, float attn_drop, float proj_drop, int sr_ratio)
    : dim(dim), num_heads(num_heads), sr_ratio(sr_ratio) {
    int head_dim = dim / num_heads;
    float scale = (float)pow(head_dim, -0.5);

    q = register_module("q", nn::Linear(nn::LinearOptions(dim, dim).bias(qkv_bias)));
    kv = register_module("kv", nn::Linear(nn::LinearOptions(dim, dim * 2).bias(qkv_bias)));
    attn_drop_ = register_module("attn_drop", nn::Dropout(nn::DropoutOptions(attn_drop)));
    proj = register_module("proj", nn::Linear(nn::LinearOptions(dim, dim)));
    proj_drop_ = register_module("proj_drop", nn::Dropout(nn::DropoutOptions(proj_drop)));
    if (sr_ratio > 1) {
        sr = register_module(
                "sr", nn::Conv2d(nn::Conv2dOptions(dim, dim, sr_ratio).stride(sr_ratio)));
        norm = register_module("norm", nn::LayerNorm(nn::LayerNormOptions({dim})));
    }
}

Tensor AttentionImpl::forward(Tensor x, int H, int W) {
    int B = x.size(0);
    int N = x.size(1);
    int C = x.size(2);

    Tensor q_ = q(x).reshape({B, N, num_heads, C / num_heads}).permute({0, 2, 1, 3});
    Tensor kv_;
    if (sr_ratio > 1) {
        Tensor x_ = x.permute({0, 2, 1}).reshape({B, C, H, W});
        x_ = sr(x_).reshape({B, C, -1}).permute({0, 2, 1});
        x_ = norm(x_);
        kv_ = kv(x_).reshape({B, -1, 2, num_heads, C / num_heads}).permute({2, 0, 3, 1, 4});
    } else {
        kv_ = kv(x).reshape({B, -1, 2, num_heads, C / num_heads}).permute({2, 0, 3, 1, 4});
    }

    Tensor k = kv_[0];
    Tensor v = kv_[1];
    Tensor attn = matmul(q_, k.transpose(-2, -1)) * scale;
    attn = softmax(attn, -1);
    attn = attn_drop_(attn);

    x = matmul(attn, v).transpose(1, 2).reshape({B, N, C});
    x = proj(x);
    x = proj_drop_(x);
    return x;
}

DWConvImpl::DWConvImpl(int dim) : dim(dim) {
    dwconv = register_module(
            "dwconv", nn::Conv2d(nn::Conv2dOptions(dim, dim, 3).stride(1).padding(1).groups(dim)));
}

Tensor DWConvImpl::forward(Tensor x, int H, int W) {
    int B = x.size(0);
    int N = x.size(1);
    int C = x.size(2);

    x = x.transpose(1, 2).view({B, C, H, W});
    x = dwconv(x);
    x = x.flatten(2).transpose(1, 2);
    return x;
}

MixFFNImpl::MixFFNImpl(int in_features, int hidden_features, float drop) {
    int out_features = in_features;
    fc1 = register_module("fc1", nn::Linear(nn::LinearOptions(in_features, hidden_features)));
    dwconv = register_module("dwconv", DWConv(hidden_features));
    fc2 = register_module("fc2", nn::Linear(nn::LinearOptions(hidden_features, out_features)));
    drop_ = register_module("drop", nn::Dropout(nn::DropoutOptions(drop)));
}

Tensor MixFFNImpl::forward(Tensor x, int H, int W) {
    x = fc1(x);
    x = nn::functional::gelu(dwconv(x, H, W));
    x = drop_(x);
    x = fc2(x);
    x = drop_(x);
    return x;
}

Tensor DropPathImpl::forward(const Tensor& x) {
    if (drop_prob == 0.0 || !is_training()) return x;
    float keep_prob = 1.0 - drop_prob;
    auto shape = std::vector<int64_t>(x.dim(), 1);
    shape[0] = x.size(0);  // only vary along batch dimension
    auto random_tensor = keep_prob + torch::rand(shape, x.options());  // same dtype/device as x
    random_tensor.floor_();                                            // binarize (0 or 1)
    return x.div(keep_prob) * random_tensor;
}

BlockImpl::BlockImpl(
        int dim,
        int num_heads,
        float mlp_ratio,
        bool qkv_bias,
        float drop,
        float attn_drop,
        float drop_path,
        int sr_ratio) {
    norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({dim})));
    attn = register_module("attn", Attention(dim, num_heads, qkv_bias, attn_drop, drop, sr_ratio));
    drop_path_ = register_module("drop_path", DropPath(drop_path));
    norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({dim})));
    int mlp_hidden_dim = (int)(dim * mlp_ratio);
    mlp = register_module("mlp", MixFFN(dim, mlp_hidden_dim, drop));
}

Tensor BlockImpl::forward(Tensor x, int H, int W) {
    x = x + drop_path_(attn(norm1(x), H, W));
    x = x + drop_path_(mlp(norm1(x), H, W));
    return x;
}

MixVisionTransformerImpl::MixVisionTransformerImpl(
        int img_size,
        int in_chans,
        std::vector<int> embed_dims,
        std::vector<int> num_heads,
        std::vector<float> mlp_ratios,
        bool qkv_bias,
        float drop_rate,
        float attn_drop_rate,
        float drop_path_rate,
        std::vector<int> depths,
        std::vector<int> sr_ratios) {
    patch_embed1 = register_module(
            "patch_embed1",
            OverlapPatchEmbed(img_size, /*patch_size*/ 7, /*stride*/ 4, in_chans, embed_dims[0]));
    patch_embed2 = register_module(
            "patch_embed2",
            OverlapPatchEmbed(
                    img_size / 4, /*patch_size*/ 3, /*stride*/ 2, embed_dims[0], embed_dims[1]));
    patch_embed3 = register_module(
            "patch_embed3",
            OverlapPatchEmbed(
                    img_size / 8, /*patch_size*/ 3, /*stride*/ 2, embed_dims[1], embed_dims[2]));
    patch_embed4 = register_module(
            "patch_embed4",
            OverlapPatchEmbed(
                    img_size / 16, /*patch_size*/ 3, /*stride*/ 2, embed_dims[2], embed_dims[3]));

    int total_depth = std::accumulate(depths.begin(), depths.end(), 0);
    torch::Tensor lin = torch::linspace(
            0, drop_path_rate, total_depth, torch::TensorOptions().dtype(torch::kFloat32));

    std::vector<float> dpr(total_depth);
    for (int i = 0; i < total_depth; i++) {
        dpr[i] = lin[i].item<float>();
    }

    int cur = 0;
    for (int i = 0; i < depths[0]; i++) {
        auto block = Block(
                /*dim*/ embed_dims[0],
                /*num_heads*/ num_heads[0],
                mlp_ratios[0],
                qkv_bias,
                drop_rate,
                attn_drop_rate,
                dpr[cur + i],
                sr_ratios[0]);
        blocks1.push_back(block);
        register_module("block_" + std::to_string(cur + i), block);
    }
    norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({embed_dims[0]})));

    cur += depths[0];
    for (int i = 0; i < depths[1]; i++) {
        auto block = Block(
                /*dim*/ embed_dims[1],
                /*num_heads*/ num_heads[1],
                mlp_ratios[1],
                qkv_bias,
                drop_rate,
                attn_drop_rate,
                dpr[cur + i],
                sr_ratios[1]);
        blocks2.push_back(block);
        register_module("block_" + std::to_string(cur + i), block);
    }
    norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({embed_dims[1]})));

    cur += depths[1];
    for (int i = 0; i < depths[2]; i++) {
        auto block = Block(
                /*dim*/ embed_dims[2],
                /*num_heads*/ num_heads[2],
                mlp_ratios[2],
                qkv_bias,
                drop_rate,
                attn_drop_rate,
                dpr[cur + i],
                sr_ratios[2]);
        blocks3.push_back(block);
        register_module("block_" + std::to_string(cur + i), block);
    }
    norm3 = register_module("norm3", nn::LayerNorm(nn::LayerNormOptions({embed_dims[2]})));

    cur += depths[2];
    for (int i = 0; i < depths[3]; i++) {
        auto block = Block(
                /*dim*/ embed_dims[3],
                /*num_heads*/ num_heads[3],
                mlp_ratios[3],
                qkv_bias,
                drop_rate,
                attn_drop_rate,
                dpr[cur + i],
                sr_ratios[3]);
        blocks4.push_back(block);
        register_module("block_" + std::to_string(cur + i), block);
    }
    norm4 = register_module("norm4", nn::LayerNorm(nn::LayerNormOptions({embed_dims[3]})));
}

std::vector<Tensor> MixVisionTransformerImpl::forward(Tensor x) {
    int B = x.size(0);
    int H, W;
    std::vector<Tensor> outs;

    std::tuple<torch::Tensor, int, int> out = patch_embed1(x);
    x = std::get<0>(out);
    H = std::get<1>(out);
    W = std::get<2>(out);

    for (int i = 0; i < blocks1.size(); i++) x = blocks1[i](x, H, W);

    x = norm1(x);
    x = x.reshape({B, H, W, -1}).permute({0, 3, 1, 2}).contiguous();
    outs.push_back(x);

    out = patch_embed2(x);
    x = std::get<0>(out);
    H = std::get<1>(out);
    W = std::get<2>(out);

    for (int i = 0; i < blocks2.size(); i++) x = blocks2[i](x, H, W);

    x = norm2(x);
    x = x.reshape({B, H, W, -1}).permute({0, 3, 1, 2}).contiguous();
    outs.push_back(x);

    out = patch_embed3(x);
    x = std::get<0>(out);
    H = std::get<1>(out);
    W = std::get<2>(out);

    for (int i = 0; i < blocks3.size(); i++) x = blocks3[i](x, H, W);

    x = norm3(x);
    x = x.reshape({B, H, W, -1}).permute({0, 3, 1, 2}).contiguous();
    outs.push_back(x);

    out = patch_embed4(x);
    x = std::get<0>(out);
    H = std::get<1>(out);
    W = std::get<2>(out);

    for (int i = 0; i < blocks4.size(); i++) x = blocks4[i](x, H, W);

    x = norm4(x);
    x = x.reshape({B, H, W, -1}).permute({0, 3, 1, 2}).contiguous();
    outs.push_back(x);

    return outs;
}

MixVisionTransformerMnistImpl::MixVisionTransformerMnistImpl(
        int img_size,
        int in_chans,
        std::vector<int> embed_dims,
        std::vector<int> num_heads,
        std::vector<float> mlp_ratios,
        bool qkv_bias,
        float drop_rate,
        float attn_drop_rate,
        float drop_path_rate,
        std::vector<int> depths,
        std::vector<int> sr_ratios) {
    patch_embed1 = register_module(
            "patch_embed1",
            OverlapPatchEmbed(img_size, /*patch_size*/ 7, /*stride*/ 4, in_chans, embed_dims[0]));
    patch_embed2 = register_module(
            "patch_embed2",
            OverlapPatchEmbed(
                    img_size / 4, /*patch_size*/ 3, /*stride*/ 2, embed_dims[0], embed_dims[1]));

    int total_depth = std::accumulate(depths.begin(), depths.end(), 0);
    torch::Tensor lin = torch::linspace(
            0, drop_path_rate, total_depth, torch::TensorOptions().dtype(torch::kFloat32));

    std::vector<float> dpr(total_depth);
    for (int i = 0; i < total_depth; i++) {
        dpr[i] = lin[i].item<float>();
    }

    int cur = 0;
    for (int i = 0; i < depths[0]; i++) {
        auto block = Block(
                /*dim*/ embed_dims[0],
                /*num_heads*/ num_heads[0],
                mlp_ratios[0],
                qkv_bias,
                drop_rate,
                attn_drop_rate,
                dpr[cur + i],
                sr_ratios[0]);
        blocks1.push_back(block);
        register_module("block_" + std::to_string(cur + i), block);
    }
    norm1 = register_module("norm1", nn::LayerNorm(nn::LayerNormOptions({embed_dims[0]})));

    cur += depths[0];
    for (int i = 0; i < depths[1]; i++) {
        auto block = Block(
                /*dim*/ embed_dims[1],
                /*num_heads*/ num_heads[1],
                mlp_ratios[1],
                qkv_bias,
                drop_rate,
                attn_drop_rate,
                dpr[cur + i],
                sr_ratios[1]);
        blocks2.push_back(block);
        register_module("block_" + std::to_string(cur + i), block);
    }
    norm2 = register_module("norm2", nn::LayerNorm(nn::LayerNormOptions({embed_dims[1]})));
}

std::vector<Tensor> MixVisionTransformerMnistImpl::forward(Tensor x) {
    int B = x.size(0);
    int H, W;
    std::vector<Tensor> outs;

    std::tuple<torch::Tensor, int, int> out = patch_embed1(x);
    x = std::get<0>(out);
    H = std::get<1>(out);
    W = std::get<2>(out);

    for (int i = 0; i < blocks1.size(); i++) x = blocks1[i](x, H, W);

    x = norm1(x);
    x = x.reshape({B, H, W, -1}).permute({0, 3, 1, 2}).contiguous();
    outs.push_back(x);

    out = patch_embed2(x);
    x = std::get<0>(out);
    H = std::get<1>(out);
    W = std::get<2>(out);

    for (int i = 0; i < blocks2.size(); i++) x = blocks2[i](x, H, W);

    x = norm2(x);
    x = x.reshape({B, H, W, -1}).permute({0, 3, 1, 2}).contiguous();
    outs.push_back(x);

    return outs;
}

}  // namespace ddpm

#endif