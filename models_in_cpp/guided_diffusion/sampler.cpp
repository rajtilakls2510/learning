
#include <torch/torch.h>

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "diffusion_utils.h"
#include "network.h"

using namespace ddpm;

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

    // std::vector<int> cm = {2, 4, 4};
    // auto model = unet::UNetModel(
    //         /*img size*/ 28,
    //         /*in channels*/ 1,
    //         /*model channels*/ 64,
    //         /*out channels*/ 1,
    //         /*num res blocks*/ 1,
    //         /*dropout*/ 0.1,
    //         /*num heads*/ 4,
    //         /*begin attention after level*/ 0,
    //         /*channel multipliers*/ cm);
    auto model = unet::SimpleUNet(
            /*img_size*/ 28,
            /*img_channels*/ 1,
            /*time_dim*/ 256,
            /*channel_dims*/ std::vector<int>{128, 512, 512});
    torch::load(model, checkpoint_path + "/model.pth");
    model->to(device);
    model->eval();

    // noise schedule
    // torch::Tensor betas = linear_schedule(max_diffusion_time).to(device);
    torch::Tensor betas = cosine_beta_schedule(max_diffusion_time).to(device);
    torch::Tensor alphas = 1.0 - betas;
    torch::Tensor alphas_cumprod = torch::cumprod(alphas, 0);
    torch::Tensor alphas_cumprod_prev =
            torch::cat({torch::ones({1}).to(device), alphas_cumprod.index({Slice(0, -1)})});
    torch::Tensor sqrt_one_minus_alphas_cumprod = torch::sqrt(1.0 - alphas_cumprod);
    torch::Tensor beta_tildes = ((1 - alphas_cumprod_prev) * betas) / (1.0 - alphas_cumprod);

    torch::NoGradGuard no_grad;
    int batch_size = 8;  // show 8 samples at once

    torch::Tensor x = torch::randn({batch_size, 1, 28, 28}).to(device);

    cv::namedWindow("Diffusion", cv::WINDOW_AUTOSIZE);

    // Before loop: create video writer
    auto now = std::chrono::system_clock::now();
    auto epoch_seconds =
            std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    std::string video_path =
            checkpoint_path + "/diffusion_" + std::to_string(epoch_seconds) + ".mp4";

    int fps = 60;  // frames per second
    cv::VideoWriter writer;
    bool isWriterOpen = false;

    for (int t = max_diffusion_time; t > 0; t--) {
        torch::Tensor timesteps = torch::full(
                {batch_size}, t, torch::TensorOptions().device(device).dtype(torch::kLong));
        torch::Tensor z = torch::randn_like(x).to(device);
        if (t == 1) z = torch::zeros_like(x).to(device);

        auto predicted_noise = model->forward(x, timesteps);

        torch::Tensor sqrt_alpha_t = torch::sqrt(extract(alphas, timesteps - 1, x.sizes()));
        torch::Tensor sqrt_one_minus_alphas_cumprod_t =
                extract(sqrt_one_minus_alphas_cumprod, timesteps - 1, x.sizes());
        torch::Tensor beta_t = extract(betas, timesteps - 1, x.sizes());

        x = (1.0 / sqrt_alpha_t) * (x - beta_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t);

        torch::Tensor sqrt_beta_tilde_t =
                torch::sqrt(extract(beta_tildes, timesteps - 1, x.sizes()));
        x += sqrt_beta_tilde_t * z;

        // Clamp and scale to [0,1]
        auto x_vis = torch::clamp(x, -1.0, 1.0);
        x_vis = (x_vis + 1) / 2;

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

    cv::destroyAllWindows();
    return 0;
}
