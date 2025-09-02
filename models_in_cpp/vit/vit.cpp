#ifndef VIT_CPP
#define VIT_CPP

#include "vit.h"

namespace vit {

    torch::Tensor posemb_sincos_2d(int h, int w, int dim, int temp){
        std::vector<torch::Tensor> o = torch::meshgrid({torch::arange(h, torch::dtype(torch::kLong)), torch::arange(w, torch::dtype(torch::kLong))}, "ij");
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

        ViTImpl::ViTImpl(int img_size, int patch_size, int num_classes, int dim, int depth, int heads, int mlp_dim, int channels) : patch_size(patch_size) {
            int patch_dim = channels * patch_size * patch_size;
            to_patch_embedding = torch::nn::Sequential();
            to_patch_embedding->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({patch_dim})));
            to_patch_embedding->push_back(torch::nn::Linear(patch_dim, dim));
            to_patch_embedding->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
            register_module("to_patch_embedding", to_patch_embedding); 

            pos_embedding = posemb_sincos_2d(img_size / patch_size, img_size / patch_size, dim);
            auto cls_pos = torch::zeros({1, 1, dim}, pos_embedding.options());
            pos_embedding = torch::cat({cls_pos, pos_embedding}, 1);

            // Register buffer
            pos_embedding = register_buffer("pos_embedding", pos_embedding);

            auto encoder_layer = torch::nn::TransformerEncoderLayer(
                torch::nn::TransformerEncoderLayerOptions(dim, heads)
                .dim_feedforward(mlp_dim)
                .activation(torch::kGELU)
            );
            encoder = register_module("encoder", torch::nn::TransformerEncoder(encoder_layer, depth));

            class_head = torch::nn::Sequential();
            class_head->push_back(torch::nn::Linear(dim, dim));
            class_head->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
            class_head->push_back(torch::nn::Linear(dim, dim));
            class_head->push_back(torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
            class_head->push_back(torch::nn::Linear(dim, num_classes));
            register_module("class_head", class_head);            

        }

        torch::Tensor ViTImpl::forward(torch::Tensor x) {
            x = patchify(x, patch_size, patch_size);
            auto class_token = torch::zeros({x.size(0), 1, x.size(2)}, x.options());
            x = torch::cat({class_token, x}, 1);
            
            x = to_patch_embedding->forward(x);
            x += pos_embedding;

            // Permute to (seq_len, batch, embed_dim)
            x = x.permute({1, 0, 2});
            x = encoder->forward(x);
            x = x.index({0}); 
            x = class_head->forward(x);
            return x;
        }

}   // namespace vitt

#endif