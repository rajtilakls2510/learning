#ifndef NETWORK_CPP
#define NETWORK_CPP

#include "network.h"

namespace ddpm {

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
        int time_depth,
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

    embedding_time = register_module(
            "embedding_time",
            torch::nn::Embedding(torch::nn::EmbeddingOptions(max_diffusion_time + 1, dim)));
    auto encoder_layer =
            torch::nn::TransformerEncoderLayer(torch::nn::TransformerEncoderLayerOptions(dim, heads)
                                                       .dim_feedforward(mlp_dim)
                                                       .activation(torch::kGELU));
    time_encoder = register_module(
            "time_encoder",
            torch::nn::TransformerEncoder(
                    torch::nn::TransformerEncoderOptions(encoder_layer, time_depth)));

    auto decoder_layer =
            torch::nn::TransformerDecoderLayer(torch::nn::TransformerDecoderLayerOptions(dim, heads)
                                                       .dim_feedforward(mlp_dim)
                                                       .activation(torch::kGELU));
    decoder = register_module("decoder", torch::nn::TransformerDecoder(decoder_layer, depth));

    reconstruction_head = torch::nn::Linear(dim, patch_size * patch_size * channels);
    register_module("reconstruction_head", reconstruction_head);
}

torch::Tensor ViTImpl::forward(torch::Tensor x, torch::Tensor t) {
    x = patchify(x, patch_size, patch_size);

    t = embedding_time(t).permute({1, 0, 2});  // [1, batch, dim]
    t = time_encoder(t);                       // [1, batch, dim]

    x = to_patch_embedding->forward(x);
    x += pos_embedding;

    // Permute to (seq_len, batch, embed_dim)
    x = x.permute({1, 0, 2});
    x = decoder->forward(/* target */ x, /* memory */ t);

    x = x.permute({1, 0, 2});  // back to [batch, seq_len, embed_dim]

    // Map back to patches
    x = reconstruction_head->forward(x);

    // Depatchify
    x = depatchify(x, patch_size, img_size, img_size, channels);
    return x;
}

}  // namespace ddpm

#endif