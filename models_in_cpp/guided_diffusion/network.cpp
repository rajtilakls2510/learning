#ifndef NETWORK_CPP
#define NETWORK_CPP

#include "network.h"

namespace ddpm::unet {

Tensor timestep_embedding(Tensor t, int dim) {
    // t : [B,]
    int half_dim = dim / 2;
    torch::Tensor scale = torch::log(torch::tensor((float)10000)) / (half_dim - 1);
    torch::Tensor exponents = torch::exp(torch::arange(half_dim).to(t.device()) * -scale);
    torch::Tensor args = t.unsqueeze(-1) * exponents.unsqueeze(0);
    return torch::cat({args.sin(), args.cos()}, /*dim*/ -1);  // [B, dim]
}

UpsampleImpl::UpsampleImpl(int channels, bool use_conv, int out_channels) : use_conv(use_conv) {
    if (use_conv)
        conv = register_module(
                "conv", Conv2d(Conv2dOptions(out_channels, out_channels, 3).padding(1)));
    bn = register_module("bn", BatchNorm2d(BatchNorm2dOptions(out_channels)));
    spatial = register_module(
            "spatial",
            ConvTranspose2d(
                    ConvTranspose2dOptions(channels, out_channels, 4).stride(2).padding(1)));
}

Tensor UpsampleImpl::forward(Tensor x) {
    x = spatial(x);
    if (use_conv) x = conv(x);
    return bn(functional::relu(x));
}

DownsampleImpl::DownsampleImpl(int channels, bool use_conv, int out_channels) : use_conv(use_conv) {
    bn = register_module("bn", BatchNorm2d(BatchNorm2dOptions(out_channels)));
    // if (use_conv)
    conv = register_module(
            "conv", Conv2d(Conv2dOptions(channels, out_channels, 3).stride(2).padding(1)));
    // else
    //     pool = register_module("pool", AvgPool2d(AvgPool2dOptions({2, 2}).stride({2, 2})));
}

Tensor DownsampleImpl::forward(Tensor x) {
    // x = use_conv ? conv(x) : pool(x);
    x = conv(x);
    return bn(functional::relu(x));
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
    norm = register_module("norm", GroupNorm(GroupNormOptions(32, channels)));
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
    Tensor qkv_ = qkv(norm(x_));
    Tensor h = proj_out(attention(qkv_));
    return (x_ + h).reshape(x.sizes());
}

ResBlockImpl::ResBlockImpl(
        int channels,
        int emb_channels,
        float drop,
        int out_channels,
        bool use_conv,
        bool up,
        bool down)
    : up(up), down(down), channels(channels), out_channels(out_channels) {
    up_down = up | down;
    in_layers = Sequential();
    in_layers->push_back(GroupNorm(GroupNormOptions(32, channels)));
    in_layers->push_back(SiLU());
    register_module("in_layers", in_layers);

    in_conv =
            register_module("in_conv", Conv2d(Conv2dOptions(channels, out_channels, 3).padding(1)));

    if (up) {
        h_up = register_module("h_up", Upsample(channels, false, channels));
        x_up = register_module("x_up", Upsample(channels, false, channels));
    } else {
        h_down = register_module("h_down", Downsample(channels, true, channels));
        x_down = register_module("x_down", Downsample(channels, true, channels));
    }

    emb_layers = Sequential();
    // emb_layers->push_back(BatchNorm1d(emb_channels));
    emb_layers->push_back(SiLU());
    emb_layers->push_back(Linear(LinearOptions(emb_channels, emb_channels)));
    // emb_layers->push_back(BatchNorm1d(emb_channels));
    emb_layers->push_back(SiLU());
    emb_layers->push_back(Linear(LinearOptions(emb_channels, 2 * out_channels)));
    register_module("emb_layers", emb_layers);

    out_norm = register_module("out_norm", GroupNorm(GroupNormOptions(32, out_channels)));

    out_layers = Sequential();
    out_layers->push_back(BatchNorm2d(out_channels));
    out_layers->push_back(SiLU());
    out_layers->push_back(Dropout(DropoutOptions(drop)));
    out_layers->push_back(Conv2d(Conv2dOptions(out_channels, out_channels, 3).padding(1)));
    register_module("out_layers", out_layers);

    if (use_conv)
        skip_connection = register_module(
                "skip_conn", Conv2d(Conv2dOptions(channels, out_channels, 3).padding(1)));
    else
        skip_connection =
                register_module("skip_conn", Conv2d(Conv2dOptions(channels, out_channels, 1)));
    skip_bn = register_module("skip_bn", BatchNorm2d(out_channels));
}

Tensor ResBlockImpl::forward(Tensor x, Tensor emb) {
    Tensor h = in_layers->forward(x);
    if (up_down) {
        h = up ? h_up(h) : h_down(h);
        x = up ? x_up(x) : x_down(x);
    }
    h = in_conv(h);

    Tensor emb_out = emb_layers->forward(emb);
    while (emb_out.sizes().size() < h.sizes().size()) emb_out = emb_out.unsqueeze(-1);

    auto chunks = emb_out.chunk(2, /*dim*/ 1);
    Tensor scale = chunks[0];
    Tensor shift = chunks[1];
    h = out_norm(h) * (1 + scale) + shift;
    h = out_layers->forward(h);

    if (channels != out_channels) h = skip_connection(x) + h;
    h = skip_bn(functional::relu(h));
    return h;
}

UNetBlockDownImpl::UNetBlockDownImpl(
        int in_channels,
        int time_embed_dim,
        int num_heads,
        int num_res_blocks,
        int out_channels,
        float dropout,
        bool use_attn,
        bool use_down)
    : use_attn(use_attn), use_down(use_down), num_res_blocks(num_res_blocks) {
    for (int i = 0; i < num_res_blocks; i++) {
        auto res_no_down = register_module(
                "res_no_down_" + std::to_string(i),
                ResBlock(
                        i == 0 ? in_channels : out_channels,
                        time_embed_dim,
                        dropout,
                        out_channels));
        res_no_downs.push_back(res_no_down);
        if (use_attn) {
            auto attn = register_module(
                    "attn_" + std::to_string(i), AttentionBlock(out_channels, num_heads));
            attns.push_back(attn);
        }
    }

    if (use_down)
        res_down = register_module(
                "res_down",
                ResBlock(
                        out_channels,
                        time_embed_dim,
                        dropout,
                        out_channels,
                        false,
                        false,
                        /*down*/ true));
}

Tensor UNetBlockDownImpl::forward(Tensor x, Tensor t) {
    for (int i = 0; i < num_res_blocks; i++) {
        x = res_no_downs[i](x, t);
        if (use_attn) x = attns[i](x);
    }
    if (use_down) x = res_down(x, t);
    return x;
}

UNetBlockUpImpl::UNetBlockUpImpl(
        int in_channels,
        int time_embed_dim,
        int num_heads,
        int num_res_blocks,
        int out_channels,
        float dropout,
        bool use_attn,
        bool use_up)
    : use_attn(use_attn), use_up(use_up), num_res_blocks(num_res_blocks) {
    for (int i = 0; i < num_res_blocks; i++) {
        auto res_no_up = register_module(
                "res_no_up_" + std::to_string(i),
                ResBlock(
                        i == 0 ? in_channels : out_channels,
                        time_embed_dim,
                        dropout,
                        out_channels));
        res_no_ups.push_back(res_no_up);
        if (use_attn) {
            auto attn = register_module(
                    "attn_" + std::to_string(i), AttentionBlock(out_channels, num_heads));
            attns.push_back(attn);
        }
    }
    if (use_up)
        res_up = register_module(
                "res_up",
                ResBlock(
                        out_channels,
                        time_embed_dim,
                        dropout,
                        out_channels,
                        false,
                        /*up*/ true,
                        false));
}

Tensor UNetBlockUpImpl::forward(Tensor x, Tensor t) {
    for (int i = 0; i < num_res_blocks; i++) {
        x = res_no_ups[i](x, t);
        if (use_attn) x = attns[i](x);
    }
    if (use_up) x = res_up(x, t);
    return x;
}

UNetModelImpl::UNetModelImpl(
        int img_size,
        int in_channels,
        int model_channels,
        int out_channels,
        int num_res_blocks,
        float dropout,
        int num_heads,
        int begin_attn,
        std::vector<int> channel_mult)
    : model_channels(model_channels), num_res_blocks(num_res_blocks), channel_mult(channel_mult) {
    int time_embed_dim = model_channels * 4;
    time_embed = Sequential();
    time_embed->push_back(Linear(model_channels, time_embed_dim));
    // time_embed->push_back(BatchNorm1d(time_embed_dim));
    time_embed->push_back(SiLU());
    register_module("time_embed", time_embed);

    int in_channel = model_channels * channel_mult[0];
    inp = register_module("inp", Conv2d(Conv2dOptions(in_channels, in_channel, 3).padding(1)));

    std::vector<int> input_block_channels = {in_channel};
    for (int level = 0; level < channel_mult.size(); level++) {
        int out_channel = model_channels * channel_mult[level];

        auto down_block = register_module(
                "down_block_" + std::to_string(level),
                UNetBlockDown(
                        in_channel,
                        time_embed_dim,
                        num_heads,
                        num_res_blocks,
                        out_channel,
                        dropout,
                        level > begin_attn,
                        level < (int)channel_mult.size() - 1));
        down_blocks.push_back(down_block);
        in_channel = out_channel;
        input_block_channels.push_back(in_channel);
    }

    int ch = in_channel;

    mid_res1 = register_module("mid_res_1", ResBlock(ch, time_embed_dim, dropout, ch));
    mid_attn = register_module("in_attn", AttentionBlock(ch, num_heads));
    mid_res2 = register_module("mid_res_2", ResBlock(ch, time_embed_dim, dropout, ch));

    for (int level = (int)channel_mult.size() - 1; level >= 0; level--) {
        int last = input_block_channels.back();
        input_block_channels.pop_back();
        in_channel = in_channel + last;
        int out_channel = model_channels * channel_mult[(level - 1) < 0 ? 0 : level - 1];

        auto up_block = register_module(
                "up_block_" + std::to_string(level),
                UNetBlockUp(
                        in_channel,
                        time_embed_dim,
                        num_heads,
                        num_res_blocks,
                        out_channel,
                        dropout,
                        level > begin_attn,
                        level < (int)channel_mult.size() - 1));
        up_blocks.push_back(up_block);
        in_channel = out_channel;
    }

    out = Sequential();
    out->push_back(GroupNorm(GroupNormOptions(32, channel_mult[0] * model_channels)));
    out->push_back(SiLU());
    out->push_back(
            Conv2d(Conv2dOptions(channel_mult[0] * model_channels, out_channels, 3).padding(1)));
    register_module("out", out);
}

Tensor UNetModelImpl::forward(Tensor x, Tensor t) {
    std::vector<Tensor> hs;
    Tensor emb = time_embed->forward(timestep_embedding(t, model_channels));

    Tensor h = x;
    h = inp(h);

    for (int i = 0; i < down_blocks.size(); i++) {
        h = down_blocks[i](h, emb);
        hs.push_back(h);
    }

    // std::cout << "Hs sizes: \n";
    // for (auto& hh : hs) std::cout << get_size(hh) << "\n";

    h = mid_res1(h, emb);
    h = mid_attn(h);
    h = mid_res2(h, emb);

    for (int i = 0; i < up_blocks.size(); i++) {
        auto last = hs.back();
        hs.pop_back();
        h = torch::cat({h, last}, /*dim*/ 1);
        h = up_blocks[i](h, emb);
    }

    h = out->forward(h);
    return h;
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

UNetDownsampleImpl::UNetDownsampleImpl(int in_ch, int out_ch, int time_dim, int kernel_size)
    : in_ch(in_ch), out_ch(out_ch), time_dim(time_dim), kernel_size(kernel_size) {
    time_encoder = register_module("time_encoder", TimePosEncoding(time_dim));
    conv_in = register_module(
            "conv_in", Conv2d(Conv2dOptions(in_ch, out_ch, kernel_size).padding(1).bias(true)));
    spatial = register_module(
            "spatial", Conv2d(Conv2dOptions(out_ch, out_ch, 4).stride(2).padding(1).bias(true)));
    bn1 = register_module("bn1", BatchNorm2d(BatchNorm2dOptions(out_ch)));
    bn2 = register_module("bn2", BatchNorm2d(BatchNorm2dOptions(out_ch)));
    conv_feat = register_module(
            "conv_feat", Conv2d(Conv2dOptions(out_ch, out_ch, 3).padding(1).bias(true)));
    time_proj = register_module("time_proj", Linear(time_dim, out_ch));
}

torch::Tensor UNetDownsampleImpl::forward(torch::Tensor x, torch::Tensor t) {
    // x : [B, C, H, W] t : [B,]
    x = bn1(functional::relu(conv_in(x)));
    torch::Tensor t_emb = functional::relu(time_proj(time_encoder(t)));
    x += t_emb.unsqueeze(-1).unsqueeze(-1);
    x = bn2(functional::relu(conv_feat(x)));
    return spatial(x);
}

UNetUpsampleImpl::UNetUpsampleImpl(int in_ch, int out_ch, int time_dim, int kernel_size)
    : in_ch(in_ch), out_ch(out_ch), time_dim(time_dim), kernel_size(kernel_size) {
    time_encoder = register_module("time_encoder", TimePosEncoding(time_dim));
    conv_in = register_module(
            "conv_in", Conv2d(Conv2dOptions(2 * in_ch, out_ch, kernel_size).padding(1).bias(true)));
    spatial = register_module(
            "spatial",
            ConvTranspose2d(
                    ConvTranspose2dOptions(out_ch, out_ch, 4).stride(2).padding(1).bias(true)));
    bn1 = register_module("bn1", BatchNorm2d(BatchNorm2dOptions(out_ch)));
    bn2 = register_module("bn2", BatchNorm2d(BatchNorm2dOptions(out_ch)));
    conv_feat = register_module(
            "conv_feat", Conv2d(Conv2dOptions(out_ch, out_ch, 3).padding(1).bias(true)));
    time_proj = register_module("time_proj", Linear(time_dim, out_ch));
}

torch::Tensor UNetUpsampleImpl::forward(torch::Tensor x, torch::Tensor t) {
    // x : [B, C, H, W] t : [B,]
    x = bn1(functional::relu(conv_in(x)));
    torch::Tensor t_emb = functional::relu(time_proj(time_encoder(t)));
    x += t_emb.unsqueeze(-1).unsqueeze(-1);
    x = bn2(functional::relu(conv_feat(x)));
    return spatial(x);
}

SimpleUNetImpl::SimpleUNetImpl(
        int img_size, int img_channels, int time_dim, std::vector<int> channel_sequence) {
    stem = register_module(
            "stem",
            Conv2d(Conv2dOptions(img_channels, channel_sequence[0], 3).padding(1).bias(true)));

    for (int i = 1; i < channel_sequence.size(); i++) {
        auto down_block = UNetDownsample(
                channel_sequence[i - 1], channel_sequence[i], time_dim, /*kernel_size*/ 3);
        down_blocks.push_back(down_block);
        register_module("down_block_" + std::to_string(i), down_block);
    }

    for (int i = (int)channel_sequence.size() - 1; i > 0; i--) {
        auto up_block = UNetUpsample(
                channel_sequence[i], channel_sequence[i - 1], time_dim, /*kernel_size*/ 3);
        up_blocks.push_back(up_block);
        register_module("up_block_" + std::to_string(i), up_block);
    }

    head = register_module(
            "head", Conv2d(Conv2dOptions(channel_sequence[0], img_channels, 1).bias(true)));
}

torch::Tensor SimpleUNetImpl::forward(torch::Tensor x, torch::Tensor t) {
    // x : [B, img_channels, H, W], t : [B,]

    std::vector<torch::Tensor> skips;
    torch::Tensor h = stem(x);

    for (int i = 0; i < down_blocks.size(); i++) {
        h = down_blocks[i](h, t);
        skips.push_back(h);
    }

    for (int i = 0; i < up_blocks.size(); i++) {
        h = up_blocks[i](torch::cat({h, skips[(int)up_blocks.size() - i - 1]}, /*dim*/ 1), t);
    }
    return head(h);
}

}  // namespace ddpm::unet
#endif