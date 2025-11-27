
#include <torch/torch.h>

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "network.h"

using namespace vdm;

// Convert a batch of [B,1,28,28] tensors into a single OpenCV image arranged in a grid
cv::Mat batch_to_mat(torch::Tensor batch, int scale = 4, int images_per_row = 4) {
    int B = batch.size(0);
    int H = batch.size(2);
    int W = batch.size(3);

    batch = batch.squeeze(1).detach().cpu();  // [B,H,W]
    batch = batch.mul(255).clamp(0, 255).to(torch::kU8);

    int rows = (B + images_per_row - 1) / images_per_row;  // ceil(B / images_per_row)

    // Canvas dimensions
    int canvas_h = rows * H * scale;
    int canvas_w = images_per_row * W * scale;
    cv::Mat canvas(canvas_h, canvas_w, CV_8U, cv::Scalar(0));

    for (int i = 0; i < B; i++) {
        cv::Mat img(cv::Size(W, H), CV_8U, batch[i].data_ptr());
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(W * scale, H * scale), 0, 0, cv::INTER_NEAREST);

        int row = i / images_per_row;
        int col = i % images_per_row;

        // Region of interest (ROI) where the image will be placed
        cv::Rect roi(col * W * scale, row * H * scale, W * scale, H * scale);
        resized.copyTo(canvas(roi));
    }

    return canvas;
}

torch::Tensor encode(torch::Tensor x) { return 2 * x - 1; }

torch::Tensor decode(torch::Tensor z, torch::Tensor g_0) {
    torch::Tensor x_vals = torch::arange(0, 255 + 1).unsqueeze(-1) / 255.0;
    x_vals = x_vals.repeat({1, 1}).to(z.device());
    x_vals = encode(x_vals).transpose(1, 0).unsqueeze(0).unsqueeze(-2).unsqueeze(-2);
    torch::Tensor inv_stddev = torch::exp(-0.5 * g_0);
    torch::Tensor logits = -0.5 * torch::pow((z.unsqueeze(-1) - x_vals) / inv_stddev, 2);
    return torch::nn::functional::log_softmax(logits, /*dim*/ -1);
}

torch::Tensor sample(
        torch::Tensor i, int T, torch::Tensor z_t, unet::NoiseNet gamma, unet::ScoreModel model) {
    // i: [1,1], z_t: [B,C,H,W]
    torch::Tensor eps = torch::randn_like(z_t).to(z_t.device());                   // [B,C,H,W]
    auto t = (T - i) / T;                                                          // [1,1]
    auto s = (T - i - 1) / T;                                                      // [1,1]
    auto t_0 = torch::zeros({1, 1}).to(z_t.device());                              // [1,1]
    auto g_0 = gamma(t_0);                                                         // [1,1]
    auto t_1 = torch::ones({1, 1}).to(z_t.device());                               // [1,1]
    auto g_1 = gamma(t_1);                                                         // [1,1]
    auto g_t = gamma(t);                                                           // [1,1]
    auto g_s = gamma(s);                                                           // [1,1]
    auto eps_hat = model(z_t, g_t.squeeze(-1), g_0.squeeze(-1), g_1.squeeze(-1));  // [B,C,H,W]
    auto a = torch::sigmoid(-g_s);                                                 // [1,1]
    auto b = torch::sigmoid(-g_t);                                                 // [1,1]
    auto c = -torch::expm1(g_s - g_t);                                             // [1,1]
    auto sigma_t = torch::sqrt(torch::sigmoid(g_t));                               // [1,1]
    auto z_s = torch::sqrt(a / b) * (z_t - sigma_t * c * eps_hat) +
               torch::sqrt((1.0 - a) * c) * eps;  // [B,C,H,W]
    return z_s;
}

torch::Tensor generate_x(torch::Tensor z_0, unet::NoiseNet gamma) {
    // z: [B,C,H,W]
    auto t_0 = torch::zeros({1, 1}).to(z_0.device());    // [1,1]
    auto g_0 = gamma(t_0);                               // [1,1]
    auto var_0 = torch::sigmoid(g_0);                    // [1,1]
    auto z_0_rescaled = z_0 / torch::sqrt(1.0 - var_0);  // [B,C,H,W]
    auto logits = decode(z_0_rescaled, g_0);             // [B,C,H,W,VS+1]
    auto samples = torch::argmax(logits, -1);            // [B,C,H,W]
    return samples;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: ./sample <checkpoint_path>\n";
        return -1;
    }

    bool use_cpu = false;
    torch::Device device{torch::kCPU};
    if (!use_cpu)
        device = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                             : torch::Device(torch::kCPU);

    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    std::string checkpoint_path = argv[1];
    int max_diffusion_time = 1000;
    auto model = unet::ScoreModel(
            /* in_out_channels */ 1,
            /* n_res_layers */ 3,
            /* n_embed */ 256,
            max_diffusion_time);
    auto gamma = unet::NoiseNet(
            /* mid_features */ 1024,
            /* gamma_min */ -13.3,
            /* gamma_max*/ 5.0);
    torch::load(model, checkpoint_path + "/model.pth");
    torch::load(gamma, checkpoint_path + "/gamma.pth");
    model->to(device);
    model->eval();
    gamma->to(device);
    gamma->eval();

    int batch_size = 8;  // show 8 samples at once

    torch::Tensor z = torch::randn({batch_size, 1, 28, 28}).to(device);  // [B,C,H,W]

    cv::namedWindow("Diffusion", cv::WINDOW_AUTOSIZE);

    // Before loop: create video writer
    auto now = std::chrono::system_clock::now();
    auto epoch_seconds =
            std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    std::string image_path =
            checkpoint_path + "/diffusion_" + std::to_string(epoch_seconds) + ".png";
    std::string video_path =
            checkpoint_path + "/diffusion_" + std::to_string(epoch_seconds) + ".mp4";

    int fps = 60;  // frames per second
    cv::VideoWriter writer;
    bool isWriterOpen = false;

    torch::NoGradGuard no_grad;

    for (int t = 0; t < max_diffusion_time + 1; t++) {
        torch::Tensor timesteps = torch::full(
                {1, 1}, t, torch::TensorOptions().device(device).dtype(torch::kLong));  // [1,1]
        if (t == max_diffusion_time) {
            z = generate_x(z, gamma);  // [B,C,H,W]
        } else {
            z = sample(timesteps, max_diffusion_time, z, gamma, model);  // [B,C,H,W]
        }
        std::cout << "t: " << t << " z: " << z << "\n";
        z = z.detach();  // detach grad for next iteration

        // Clamp and scale to [0,1]
        torch::Tensor x_vis;
        if (t == max_diffusion_time) {
            x_vis = z / 255.0;
        } else {
            x_vis = torch::clamp(z, -1.0, 1.0);
            x_vis = (x_vis + 1) / 2;
        }
        // Convert batch to OpenCV Mat
        cv::Mat canvas = batch_to_mat(x_vis, 10);

        // Initialize writer when we know the frame size
        if (!isWriterOpen) {
            writer.open(
                    video_path,
                    cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
                    fps,
                    canvas.size(),
                    false);  // false = grayscale video
            if (!writer.isOpened()) {
                std::cerr << "Error: Could not open video writer\n";
                return -1;
            }
            isWriterOpen = true;
        }

        // Write frame to video
        writer.write(canvas);
        cv::imshow("Diffusion", canvas);

        // Adjust delay: larger delay at last steps
        int delay = 1;                        // (t % 100 == 0 || t < 20) ? 200 : 30;
        if (cv::waitKey(delay) == 27) break;  // Esc to quit
    }

    if (isWriterOpen) {
        writer.release();
        std::cout << "Video saved at: " << video_path << "\n";
    }

    // Save the final generated image
    cv::imwrite(image_path, batch_to_mat(z / 255, 10));

    std::cout << "Final image saved at: " << image_path << "\n";

    cv::destroyAllWindows();
    return 0;
}
