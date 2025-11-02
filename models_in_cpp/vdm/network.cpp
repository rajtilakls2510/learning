#ifndef NETWORK_CPP
#define NETWORK_CPP

#include "network.h"

namespace vdm::unet {

Tensor timestep_embedding(Tensor t, int dim, int max_diffusion_time) {
    // t : [B,]
    t *= max_diffusion_time;
    int half_dim = dim / 2;
    torch::Tensor scale = torch::log(torch::tensor((float)10000)) / (half_dim - 1);
    torch::Tensor exponents = torch::exp(torch::arange(half_dim).to(t.device()) * -scale);
    torch::Tensor args = t.unsqueeze(-1) * exponents.unsqueeze(0);
    return torch::cat({args.sin(), args.cos()}, /*dim*/ -1);  // [B, dim]
}

DenseMonotoneImpl::DenseMonotoneImpl(int in_features, int out_features, bool use_bias)
    : use_bias(use_bias) {
    // Initialize unconstrained weight parameter
    weight = register_parameter("weight", torch::randn({in_features, out_features}) * 0.1);
    if (use_bias) bias = register_parameter("bias", torch::zeros({out_features}));
}

Tensor DenseMonotoneImpl::forward(Tensor x) {
    Tensor y = matmul(x, torch::abs(weight));
    if (use_bias) y = y + bias;
    return y;
}

NoiseNetImpl::NoiseNetImpl(int mid_features, double gamma_min, double gamma_max)
    : mid_features(mid_features), gamma_min(gamma_min), gamma_max(gamma_max) {
    double init_bias = gamma_min;
    double init_scale = gamma_max - gamma_min;
    l1 = register_module("l1", DenseMonotone(1, 1));
    l1->weight.data().fill_(init_scale);
    l1->bias.data().fill_(init_bias);
    l2 = register_module("l2", DenseMonotone(1, mid_features));
    l3 = register_module("l3", DenseMonotone(mid_features, 1, /* bias */ false));
}

Tensor NoiseNetImpl::forward(Tensor x) {
    Tensor h = l1(x);
    Tensor _h = 2 * (x - 0.5);
    _h = l2(_h);
    _h = 2 * (sigmoid(_h) - 0.5);
    _h = l3(_h) / mid_features;
    return h + _h;
}

ResnetBlockImpl::ResnetBlockImpl(int in_channels, int out_channels, int cond_dim)
    : in_channels(in_channels), out_channels(out_channels) {
    conv1 = register_module(
            "conv1", Conv2d(Conv2dOptions(in_channels, in_channels, 3).stride(1).padding(1)));
    conv2 = register_module(
            "conv2", Conv2d(Conv2dOptions(in_channels, out_channels, 3).stride(1).padding(1)));
    conv2->weight.data().zero_();

    cond_proj =
            register_module("cond_proj", Linear(LinearOptions(cond_dim, in_channels).bias(false)));
    cond_proj->weight.data().zero_();

    drop = register_module("drop", Dropout(DropoutOptions(0.1)));
    gpn1 = register_module("gpn1", GroupNorm(GroupNormOptions(32, in_channels)));
    gpn2 = register_module("gpn2", GroupNorm(GroupNormOptions(32, in_channels)));
    if (in_channels != out_channels)
        min_shortcut = register_module(
                "min_shortcut",
                Conv2d(Conv2dOptions(in_channels, out_channels, 3).stride(1).padding(1)));
}

Tensor ResnetBlockImpl::forward(Tensor x, Tensor cond) {
    Tensor h = functional::silu(gpn1(x));
    h = conv1(h);
    cond = cond_proj(cond);
    cond = cond.unsqueeze(-1).unsqueeze(-1);
    h += cond;
    h = functional::silu(gpn2(h));
    h = drop(h);
    h = conv2(h);
    if (in_channels != out_channels) x = min_shortcut(x);
    return x + h;
}

QKVAttentionImpl::QKVAttentionImpl(int nheads) : nheads(nheads) {}

Tensor QKVAttentionImpl::forward(Tensor qkv) {
    // qkv: [B x (3 * nheads * C) x T] tensor of Qs, Ks, and Vs.
    int bs = qkv.size(0);
    int width = qkv.size(1);
    int length = qkv.size(2);

    int ch = width / (3 * nheads);
    auto q_k_v_ = qkv.chunk(3, /*dim*/ 1);
    Tensor q = q_k_v_[0];
    Tensor k = q_k_v_[1];
    Tensor v = q_k_v_[2];

    float scale = (float)(1.0 / std::sqrt(std::sqrt(ch)));
    Tensor weight = torch::einsum("bct,bcs->bts", {q * scale, k * scale});
    weight = weight.softmax(/*dim*/ -1);
    Tensor a = torch::einsum("bts,bcs->bct", {weight, v});
    return a.reshape({bs, -1, length});  // [B x (nheads * C) x T]
}

AttentionBlockImpl::AttentionBlockImpl(int channels, int nheads) {
    qkv = register_module("qkv", Conv1d(Conv1dOptions(channels, 3 * channels, 1)));
    attention = register_module("attention", QKVAttention(nheads));
    proj_out = register_module("proj_out", Conv1d(Conv1dOptions(channels, channels, 1)));

    // Zero init
    init::zeros_(proj_out->weight);
    if (proj_out->options.bias()) {
        init::zeros_(proj_out->bias);
    }
}

Tensor AttentionBlockImpl::forward(Tensor x) {
    int b = x.size(0);
    int c = x.size(1);
    Tensor x_ = x.reshape({b, c, -1});
    Tensor qkv_ = qkv(x_);
    Tensor h = proj_out(attention(qkv_));
    return (x_ + h).reshape(x.sizes());
}

ScoreModelImpl::ScoreModelImpl(
        int in_out_channels,
        int n_res_layers,
        int n_embed,
        double gamma_min,
        double gamma_max,
        int max_diffusion_time)
    : n_res_layers(n_res_layers),
      n_embed(n_embed),
      gamma_min(gamma_min),
      gamma_max(gamma_max),
      max_diffusion_time(max_diffusion_time) {
    dense0 = register_module("dense0", Linear(LinearOptions(n_embed, n_embed * 4)));
    dense1 = register_module("dense1", Linear(LinearOptions(n_embed * 4, n_embed * 4)));
    conv_in = register_module(
            "conv_in", Conv2d(Conv2dOptions(in_out_channels, n_embed, 3).padding(1).stride(1)));
    for (int i = 0; i < n_res_layers; i++) {
        auto block = register_module(
                "down_block_" + std::to_string(i), ResnetBlock(n_embed, n_embed, n_embed * 4));
        down_blocks.emplace_back(block);
    }
    midres1_block = register_module("midres1_block", ResnetBlock(n_embed, n_embed, n_embed * 4));
    mid_attn_block = register_module("mid_attn_block", AttentionBlock(n_embed, 1));
    midres2_block = register_module("midres2_block", ResnetBlock(n_embed, n_embed, n_embed * 4));

    for (int i = 0; i < n_res_layers + 1; i++) {
        auto block = register_module(
                "up_block_" + std::to_string(i), ResnetBlock(n_embed * 2, n_embed, n_embed * 4));
        up_blocks.emplace_back(block);
    }
    conv_out = register_module(
            "conv_out", Conv2d(Conv2dOptions(n_embed, in_out_channels, 3).padding(1).stride(1)));
    norm = register_module("norm", GroupNorm(GroupNormOptions(32, n_embed)));
}

Tensor ScoreModelImpl::forward(Tensor z, Tensor g_t) {
    Tensor t = (g_t - gamma_min) / (gamma_max - gamma_min);
    Tensor t_emb = timestep_embedding(t, n_embed, max_diffusion_time);
    // TODO: Add conditioning
    Tensor cond = functional::silu(dense0(t_emb));
    cond = functional::silu(dense1(cond));

    std::vector<Tensor> hs;

    Tensor h = conv_in(z);
    hs.emplace_back(h);

    for (int i = 0; i < n_res_layers; i++) {
        h = down_blocks[i](h, cond);
        hs.emplace_back(h);
    }

    h = midres1_block(h, cond);
    h = mid_attn_block(h);
    h = midres2_block(h, cond);

    for (int i = 0; i < n_res_layers + 1; i++) {
        Tensor h_ = hs[n_res_layers - i];
        h = up_blocks[i](concat({h, h_}, 1), cond);
    }

    h = functional::silu(norm(h));
    h = conv_out(h);

    return z + h;
}

}  // namespace vdm::unet
#endif