#ifndef NETWORK_CPP
#define NETWORK_CPP

#include "network.h"

namespace ddpm {

torch::Tensor build_posemb_sincos_1d(int n, int dim, int max_period = 10000) {
    // n: maximum diffusion steps
    torch::Tensor positions = torch::arange(n, torch::kFloat32).unsqueeze(1);  // [n,1]

    auto half_dim = torch::arange(dim / 2, torch::kFloat32);  // [dim/2]
    half_dim = -torch::log(torch::tensor((float)max_period)) * half_dim / (dim / 2 - 1);
    auto freqs = torch::exp(half_dim);  // [dim/2]

    auto args = positions * freqs;                                    // [n, dim/2]
    auto emb = torch::cat({torch::sin(args), torch::cos(args)}, -1);  // [n, dim]
    return emb;                                                       // [n, dim]
}

torch::Tensor posemb_sincos_2d(int h, int w, int dim, int temp) {
    std::vector<torch::Tensor> o = torch::meshgrid(
            {torch::arange(h, torch::dtype(torch::kLong)),
             torch::arange(w, torch::dtype(torch::kLong))},
            "ij");
    torch::Tensor y = o[0];
    torch::Tensor x = o[1];

    torch::Tensor omega = torch::arange(dim / 4, torch::kFloat32) / (dim / 4 - 1);
    omega = 1.0 / torch::pow(temp, omega);

    y = y.flatten().to(torch::kFloat32).unsqueeze(1) * omega.unsqueeze(0);
    x = x.flatten().to(torch::kFloat32).unsqueeze(1) * omega.unsqueeze(0);

    auto pe = torch::cat({x.sin(), x.cos(), y.sin(), y.cos()}, 1);
    return pe.unsqueeze(0);
}

torch::Tensor patchify(torch::Tensor x, int p1, int p2) {
    auto sizes = x.sizes();
    int B = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];
    int h = H / p1;
    int w = W / p2;

    // Step 1: view into [B, C, h, p1, w, p2]
    x = x.view({B, C, h, p1, w, p2});

    // Step 2: permute → [B, h, w, p1, p2, C]
    x = x.permute({0, 2, 4, 3, 5, 1});

    // Step 3: reshape → [B, h*w, p1*p2*C]
    x = x.reshape({B, h * w, p1 * p2 * C});
    return x;
}

torch::Tensor depatchify(torch::Tensor patches, int p, int H, int W, int C) {
    int B = patches.size(0);
    int h = H / p;
    int w = W / p;

    patches = patches.view({B, h, w, p, p, C});
    patches = patches.permute({0, 5, 1, 3, 2, 4});  // [B, C, h, p, w, p]
    patches = patches.reshape({B, C, H, W});
    return patches;
}

TimePosEncodingImpl::TimePosEncodingImpl(int dim) : dim(dim) {}

torch::Tensor TimePosEncodingImpl::forward(torch::Tensor t) {
    // t : [B,]
    int half_dim = dim / 2;
    torch::Tensor scale = torch::log(torch::tensor((float)10000)) / (half_dim - 1);
    torch::Tensor exponents = torch::exp(torch::arange(half_dim).to(t.device()) * -scale);
    torch::Tensor args = t.unsqueeze(-1) * exponents.unsqueeze(0);
    return torch::cat({args.sin(), args.cos()}, /*dim*/ -1);  // [B, dim]
}

ViTBlockImpl::ViTBlockImpl(int dim, int heads, int mlp_dim, int time_dim)
    : dim(dim), heads(heads), mlp_dim(mlp_dim), time_dim(time_dim) {
    time_encoder = register_module("time_encoder", TimePosEncoding(time_dim));
    encoder = register_module(
            "encoder",
            TransformerEncoderLayer(TransformerEncoderLayerOptions(dim, heads)
                                            .dim_feedforward(mlp_dim)
                                            .dropout(0.1)));
    time_proj = register_module("time_proj", Linear(time_dim, dim));
}

torch::Tensor ViTBlockImpl::forward(torch::Tensor x, torch::Tensor t) {
    // x : [seq len, B, dim] t : [B,]
    torch::Tensor t_emb = functional::relu(time_proj(time_encoder(t)));  // [B, time_dim]
    x = encoder(x);
    x += t_emb.unsqueeze(0);  // [seq_len, B, dim]
    return x;
}

ViTImpl::ViTImpl(
        int img_size,
        int patch_size,
        int dim,
        int depth,
        int heads,
        int mlp_dim,
        int time_dim,
        int channels)
    : patch_size(patch_size), img_size(img_size), channels(channels), depth(depth) {
    int patch_dim = channels * patch_size * patch_size;
    to_patch_embedding = torch::nn::Sequential();
    to_patch_embedding->push_back(torch::nn::Linear(patch_dim, dim));
    to_patch_embedding->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    register_module("to_patch_embedding", to_patch_embedding);

    pos_embedding = posemb_sincos_2d(img_size / patch_size, img_size / patch_size, dim);
    pos_embedding = register_buffer("pos_embedding", pos_embedding);

    for (int i = 0; i < depth; i++) {
        auto block = ViTBlock(dim, heads, mlp_dim, time_dim);
        blocks.push_back(block);
        register_module("block_" + std::to_string(i), block);
    }

    reconstruction_head = torch::nn::Sequential(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})),
            torch::nn::Linear(dim, 2 * dim),
            torch::nn::ReLU(),
            torch::nn::Linear(2 * dim, patch_size * patch_size * channels));
    register_module("reconstruction_head", reconstruction_head);
}

torch::Tensor ViTImpl::forward(torch::Tensor x_, torch::Tensor t) {
    // x: [B, C, H, W] t: [B, ]
    // std::cout << "x0: " << net::get_size(x_) << "\n";
    torch::Tensor x = patchify(x_, patch_size, patch_size);
    // std::cout << "x1: " << net::get_size(x) << "\n";
    x = to_patch_embedding->forward(x);
    // std::cout << "x2: " << net::get_size(x) << "\n";
    x += pos_embedding;
    // std::cout << "x3: " << net::get_size(x) << "\n";

    x = x.permute({1, 0, 2});  // [sqe len, B, dim]
    for (int i = 0; i < depth; i++) {
        x = blocks[i](x, t);
        // std::cout << "x4"+std::to_string(i) + " : " << net::get_size(x) << "\n";
    }
    x = x.permute({1, 0, 2});  // back to [B, seq len, dim]
    // std::cout << "x5: " << net::get_size(x) << "\n";

    // Map back to patches
    x = reconstruction_head->forward(x);
    // std::cout << "x6: " << net::get_size(x) << "\n";

    // Depatchify
    x = depatchify(x, patch_size, img_size, img_size, channels);
    // std::cout << "x7: " << net::get_size(x) << "\n";
    return x;
}

}  // namespace ddpm

#endif