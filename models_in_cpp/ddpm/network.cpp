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

ViTImpl::ViTImpl(
        int img_size,
        int patch_size,
        int dim,
        int depth,
        int heads,
        int mlp_dim,
        int channels,
        int max_diffusion_time)
    : patch_size(patch_size), img_size(img_size), channels(channels) {
    int patch_dim = channels * patch_size * patch_size;
    to_patch_embedding = torch::nn::Sequential();
    to_patch_embedding->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({patch_dim})));
    to_patch_embedding->push_back(torch::nn::Linear(patch_dim, dim));
    to_patch_embedding->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    register_module("to_patch_embedding", to_patch_embedding);

    pos_embedding = posemb_sincos_2d(img_size / patch_size, img_size / patch_size, dim);
    pos_embedding = register_buffer("pos_embedding", pos_embedding);

    time_pos_embedding = build_posemb_sincos_1d(max_diffusion_time + 1, dim);
    time_pos_embedding = register_buffer("time_pos_embedding", time_pos_embedding);

    embedding_time = register_module(
            "embedding_time",
            torch::nn::Embedding(torch::nn::EmbeddingOptions(max_diffusion_time + 1, dim)));

    time_mlp = register_module(
            "time_mlp",
            torch::nn::Sequential(
                    torch::nn::Linear(dim, dim * 2),
                    torch::nn::GELU(),
                    torch::nn::Linear(dim * 2, dim)));

    auto image_encoder_layer =
            torch::nn::TransformerEncoderLayer(torch::nn::TransformerEncoderLayerOptions(dim, heads)
                                                       .dim_feedforward(mlp_dim)
                                                       .activation(torch::kGELU));
    image_encoder = register_module(
            "image_encoder", torch::nn::TransformerEncoder(image_encoder_layer, depth));

    // reconstruction_head = torch::nn::Linear(dim, patch_size * patch_size * channels);
    // register_module("reconstruction_head", reconstruction_head);
    reconstruction_head = torch::nn::Sequential(
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})),
            torch::nn::Linear(dim, dim),
            torch::nn::GELU(),
            torch::nn::Linear(dim, patch_size * patch_size * channels));
    register_module("reconstruction_head", reconstruction_head);
}

torch::Tensor ViTImpl::forward(torch::Tensor x_, torch::Tensor t) {
    torch::Tensor x = patchify(x_, patch_size, patch_size);

    x = to_patch_embedding->forward(x);
    x += pos_embedding;

    // --- Time embeddings ---
    // Learned time embedding
    auto t_emb_learned = embedding_time(t);  // [B, dim]

    // Sinusoidal time embedding
    auto t_emb_sin = time_pos_embedding.index_select(0, t);  // [B, dim]

    // Combine (add or concat; here we add)
    auto t_emb = t_emb_learned + t_emb_sin;

    t_emb = time_mlp->forward(t_emb);  // [B, dim]
    t_emb = t_emb.unsqueeze(1);        // [B,1,dim]
    t_emb = t_emb.expand({t_emb.size(0), x.size(1), t_emb.size(2)});
    x = x + t_emb;

    // Permute to (seq_len, batch, embed_dim)
    x = x.permute({1, 0, 2});
    x = image_encoder->forward(x);

    x = x.permute({1, 0, 2});  // back to [batch, seq_len, embed_dim]

    // Map back to patches
    x = reconstruction_head->forward(x);

    // Depatchify
    x = depatchify(x, patch_size, img_size, img_size, channels);
    return x;
}

}  // namespace ddpm

#endif