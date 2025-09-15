#ifndef UNET_MNIST_HPP
#define UNET_MNIST_HPP

// Minimal U-Net denoiser for MNIST using libtorch (C++).
// - Input: [B, 1, 28, 28]
// - Output: [B, 1, 28, 28] (predicting noise / or residual)
// - Includes simple time embedding (learned embedding + MLP) added to feature maps.

#include <torch/torch.h>

namespace ddpm {

// -------------------- Helpers --------------------
struct DoubleConvImpl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};

    DoubleConvImpl(int in_ch, int out_ch) {
        conv1 = register_module(
                "conv1",
                torch::nn::Conv2d(
                        torch::nn::Conv2dOptions(in_ch, out_ch, /*kernel_size=*/3).padding(1)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_ch));

        conv2 = register_module(
                "conv2",
                torch::nn::Conv2d(
                        torch::nn::Conv2dOptions(out_ch, out_ch, /*kernel_size=*/3).padding(1)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_ch));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(bn1->forward(conv1->forward(x)));
        x = torch::relu(bn2->forward(conv2->forward(x)));
        return x;
    }
};
TORCH_MODULE(DoubleConv);

struct DownImpl : torch::nn::Module {
    torch::nn::MaxPool2d pool{nullptr};
    DoubleConv conv{nullptr};

    DownImpl(int in_ch, int out_ch) {
        pool = register_module("pool", torch::nn::MaxPool2d(2));
        conv = register_module("conv", DoubleConv(in_ch, out_ch));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = pool->forward(x);
        x = conv->forward(x);
        return x;
    }
};
TORCH_MODULE(Down);

struct UpImpl : torch::nn::Module {
    torch::nn::ConvTranspose2d up{nullptr};
    DoubleConv conv{nullptr};

    UpImpl(int in_ch, int out_ch) {
        // in_ch is typically 2*channels due to concatenation
        up = register_module(
                "up",
                torch::nn::ConvTranspose2d(
                        torch::nn::ConvTranspose2dOptions(in_ch / 2, in_ch / 2, /*kernel_size=*/2)
                                .stride(2)));
        conv = register_module("conv", DoubleConv(in_ch, out_ch));
    }

    torch::Tensor forward(torch::Tensor x1, torch::Tensor x2) {
        // x1: decoder feature to be upsampled
        // x2: skip connection from encoder
        x1 = up->forward(x1);
        // in case sizes mismatch due to rounding, center-crop x2
        auto diffY = x2.size(2) - x1.size(2);
        auto diffX = x2.size(3) - x1.size(3);
        if (diffY != 0 || diffX != 0) {
            x2 = x2.slice(2, diffY / 2, diffY / 2 + x1.size(2))
                         .slice(3, diffX / 2, diffX / 2 + x1.size(3));
        }
        auto x = torch::cat({x2, x1}, 1);
        x = conv->forward(x);
        return x;
    }
};
TORCH_MODULE(Up);

struct OutConvImpl : torch::nn::Module {
    torch::nn::Conv2d conv{nullptr};
    OutConvImpl(int in_ch, int out_ch) {
        conv = register_module(
                "conv",
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, /*kernel_size=*/1)));
    }
    torch::Tensor forward(torch::Tensor x) { return conv->forward(x); }
};
TORCH_MODULE(OutConv);

// -------------------- Time embedding --------------------
struct TimeEmbeddingImpl : torch::nn::Module {
    torch::nn::Embedding emb{nullptr};
    torch::nn::Sequential mlp{nullptr};

    TimeEmbeddingImpl(int max_steps, int dim) {
        emb = register_module("emb", torch::nn::Embedding(max_steps, dim));
        mlp = register_module(
                "mlp",
                torch::nn::Sequential(
                        torch::nn::Linear(dim, dim * 4),
                        torch::nn::GELU(),
                        torch::nn::Linear(dim * 4, dim)));
    }

    torch::Tensor forward(torch::Tensor t) {
        // t: LongTensor shape [B]
        auto e = emb->forward(t);
        e = mlp->forward(e);
        return e;  // [B, dim]
    }
};
TORCH_MODULE(TimeEmbedding);

// -------------------- Small U-Net --------------------
struct UNetImpl : torch::nn::Module {
    // channels tuned for MNIST
    DoubleConv inc{nullptr};
    Down down1{nullptr}, down2{nullptr};
    DoubleConv bottleneck{nullptr};
    Up up2{nullptr}, up1{nullptr};
    OutConv outc{nullptr};

    TimeEmbedding time_emb{nullptr};
    torch::nn::Linear time_to_bottleneck{nullptr};

    UNetImpl(int in_channels = 1, int out_channels = 1, int base_c = 32, int max_time = 1000) {
        // encoder
        inc = register_module("inc", DoubleConv(in_channels, base_c));
        down1 = register_module("down1", Down(base_c, base_c * 2));
        down2 = register_module("down2", Down(base_c * 2, base_c * 4));

        // time embedding
        time_emb = register_module("time_emb", TimeEmbedding(max_time, base_c * 4));
        time_to_bottleneck =
                register_module("time_to_bottleneck", torch::nn::Linear(base_c * 4, base_c * 4));

        // bottleneck
        bottleneck = register_module("bottleneck", DoubleConv(base_c * 4, base_c * 4));

        // decoder
        up2 = register_module("up2", Up(base_c * 8, base_c * 2));
        up1 = register_module("up1", Up(base_c * 4, base_c));
        outc = register_module("outc", OutConv(base_c, out_channels));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor t) {
        // x: [B,1,28,28], t: LongTensor [B]
        auto x1 = inc->forward(x);     // [B, base_c, 28,28]
        auto x2 = down1->forward(x1);  // [B, base_c*2, 14,14]
        auto x3 = down2->forward(x2);  // [B, base_c*4, 7,7]

        // time embedding -> broadcast to spatial dims and added to bottleneck
        auto te = time_emb->forward(t);                     // [B, base_c*4]
        te = torch::relu(time_to_bottleneck->forward(te));  // [B, base_c*4]
        te = te.view({te.size(0), te.size(1), 1, 1});

        auto b = bottleneck->forward(x3);
        b = b + te;  // broadcast add

        auto u2 = up2->forward(b, x2);
        auto u1 = up1->forward(u2, x1);
        auto out = outc->forward(u1);

        // It's common to predict noise in same scale as input. No activation here.
        return out;
    }
};
TORCH_MODULE(UNet);

}  // namespace ddpm

// Example usage (C++):
// ddpm::UNet model(1,1,32,1000);
// model->to(device);
// auto out = model->forward(input, timesteps);
// Loss: mse(out, noise)  // if predicting noise

#endif