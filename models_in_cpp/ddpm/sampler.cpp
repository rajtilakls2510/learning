#include <torch/torch.h>

#include <iostream>
#include <opencv2/opencv.hpp>

#include "diffusion_utils.h"
#include "network.h"

using namespace ddpm;

// Function to display MNIST image tensor
void show_image(torch::Tensor img, const std::string& window_name) {
    // MNIST image is 1x28x28 (CHW)
    img = img.squeeze().detach().cpu();
    img = img.mul(255).clamp(0, 255).to(torch::kU8);

    cv::Mat mat(cv::Size(28, 28), CV_8U, img.data_ptr());
    cv::resize(mat, mat, cv::Size(280, 280), 0, 0, cv::INTER_NEAREST);

    cv::imshow(window_name, mat);
    cv::waitKey(2000);
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

    auto model = ViT(
            /* img_size */ 28,
            /* patch_size */ 4,
            /* embed_dim */ 256,
            /* depth */ 6,
            /* heads */ 4,
            /* mlp_dim */ 1024,
            /* n_channels */ 1,
            /* max diffusion time*/ max_diffusion_time);
    torch::load(model, checkpoint_path + "/model.pth");
    model->to(device);
    model->eval();

    // Initialize noise schedules
    torch::Tensor betas = linear_schedule(max_diffusion_time).to(device);//cosine_beta_schedule(max_diffusion_time).to(device);
    torch::Tensor alphas = 1.0 - betas;
    torch::Tensor alphas_cumprod = torch::cumprod(alphas, /*axis*/ 0);
    torch::Tensor alphas_cumprod_prev =
            torch::cat({torch::ones({1}).to(device), alphas_cumprod.index({Slice(0, -1)})});
    torch::Tensor sqrt_alphas_cumprod = torch::sqrt(alphas_cumprod);
    torch::Tensor sqrt_one_minus_alphas_cumprod = torch::sqrt(1.0 - alphas_cumprod);
    torch::Tensor beta_tildes = ((1 - alphas_cumprod_prev) * betas) / (1.0 - alphas_cumprod);

    torch::NoGradGuard no_grad;
    int batch_size = 1;
    for (int i = 0; i < 10; i++) {
        torch::Tensor x = torch::randn({batch_size, 1, 28, 28}).to(device);
        for (int t = max_diffusion_time; t > 0; t--) {
            torch::Tensor timesteps = torch::full(
                    {batch_size}, t, torch::TensorOptions().device(device).dtype(torch::kLong));
            // std::cout << "timesteps: " << timesteps << "\n";
            torch::Tensor z = torch::randn_like(x).to(device);
            if (t == 1) z = torch::zeros_like(x).to(device);

            // auto predicted_noise = model->forward(x, timesteps.unsqueeze(-1));
            auto predicted_noise = model->forward(x, timesteps);
            torch::Tensor sqrt_alpha_t = torch::sqrt(extract(alphas, timesteps - 1, x.sizes()));
            torch::Tensor sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod, timesteps - 1, x.sizes());
            torch::Tensor beta_t = extract(betas, timesteps - 1, x.sizes());

            // std::cout << "sqrt_alpha_t: " << sqrt_alpha_t << " sqrt_one_minus_alphas_cumprod_t: " << sqrt_one_minus_alphas_cumprod_t << " beta_t: " << beta_t << "\n";
            x = (1.0 / sqrt_alpha_t) * (x - beta_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t);
            // std::cout << "t: " << t << " predicted_noise: " << predicted_noise << " x: " << x << "\n";
            
            torch::Tensor sqrt_beta_tilde_t = torch::sqrt(extract(beta_tildes, timesteps - 1, x.sizes()));
            x += sqrt_beta_tilde_t * z;
            // x = torch::clamp(x, -1.0, 1.0);
            // x += torch::sqrt(beta_t) * z;
        
        }

        // std::cout << "x : " << x << "\n";
        x = torch::clamp(x, -1.0, 1.0);
        x = (x + 1) / 2; // Scale between [0,1]

        show_image(x[0], "img");
    }

    return 0;
}
