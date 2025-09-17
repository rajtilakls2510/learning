#ifndef TRAINER_CPP
#define TRAINER_CPP

#include "trainer.h"

namespace ddpm {
namespace fs = std::filesystem;
using namespace MNIST;
using namespace torch::indexing;

Trainer::Trainer(
        std::string data_path, std::string checkpoint_path, int max_diffusion_time, bool use_cpu)
    : data_path(data_path),
      checkpoint_path(checkpoint_path),
      max_diffusion_time(max_diffusion_time) {
    fs::path cp(checkpoint_path);
    if (!fs::exists(cp)) {
        std::cout << cp << " does not exist. Creating...\n";
        fs::create_directories(cp);
    }
    if (!use_cpu)
        device = torch::cuda::is_available() ? torch::Device(torch::kCUDA)
                                             : torch::Device(torch::kCPU);

    std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

    model = ViT(
            /* img_size */ 28,
            /* patch_size */ 4,
            /* embed_dim */ 512,
            /* depth */ 16,
            /* heads */ 8,
            /* mlp_dim */ 1024,
            /* time_dim */ 256,
            /* n_channels */ 1);
    optimizer = std::make_shared<torch::optim::Adam>(
            model->parameters(), torch::optim::AdamOptions(1e-4));  // Learning rate: 2e-5
    if (!fs::exists(cp / "model.pth")) {
        std::cout << "Did not find model. Saving initial model...\n";
        torch::save(model, (cp / "model.pth").string());
        torch::save(*optimizer, (cp / "optim.pth").string());
    } else {
        torch::load(model, (cp / "model.pth").string());
        torch::load(*optimizer, (cp / "optim.pth").string());
        std::cout << "Loaded model and optimizer.\n";
    }
    model->to(device);

    std::cout << "Num Parameters: " << net::count_parameters(model) << "\n";

    // Initialize noise schedules
    // betas = linear_schedule(max_diffusion_time)
    //                 .to(device);
    betas = cosine_beta_schedule(max_diffusion_time).to(device);

    alphas = 1.0 - betas;
    alphas_cumprod = torch::cumprod(alphas, /*axis*/ 0);
    alphas_cumprod_prev =
            torch::cat({torch::ones({1}).to(device), alphas_cumprod.index({Slice(0, -1)})});
    sqrt_alphas_cumprod = torch::sqrt(alphas_cumprod);
    sqrt_one_minus_alphas_cumprod = torch::sqrt(1.0 - alphas_cumprod);
}

torch::Tensor Trainer::q_sample(torch::Tensor x_start, torch::Tensor t, torch::Tensor noise) {
    return extract(sqrt_alphas_cumprod, t, x_start.sizes()) * x_start +
           extract(sqrt_one_minus_alphas_cumprod, t, x_start.sizes()) * noise;
}

void Trainer::train_step(Batch batch, double* loss) {
    model->train();
    auto images = batch.images.to(device);
    images = 2 * images - 1;  // Scale between [-1,1]
    torch::Tensor t, noise, x_noisy;
    {
        torch::NoGradGuard no_grad;
        t = torch::randint(1, max_diffusion_time + 1, {/*batch*/ batch.batch_size}, torch::kLong)
                    .to(device);
        noise = torch::randn_like(images).to(device);
        x_noisy = q_sample(images, t - 1, noise);
    }

    auto outputs = model->forward(x_noisy, t);

    torch::nn::MSELoss criterion;
    auto loss_tensor = criterion(outputs, noise);

    optimizer->zero_grad();
    loss_tensor.backward();
    optimizer->step();

    *loss = loss_tensor.item<double>();
}

void Trainer::test_step(Batch batch, double* loss) {
    model->eval();
    auto images = batch.images.to(device);
    images = 2 * images - 1;  // Scale between [-1,1]
    torch::Tensor t, noise, x_noisy;

    torch::NoGradGuard no_grad;
    t = torch::randint(1, max_diffusion_time + 1, {/*batch*/ batch.batch_size}, torch::kLong)
                .to(device);
    noise = torch::randn_like(images).to(device);
    x_noisy = q_sample(images, t - 1, noise);

    auto outputs = model->forward(x_noisy, t);

    torch::nn::MSELoss criterion;
    auto loss_tensor = criterion(outputs, noise);

    *loss = loss_tensor.item<double>();
}

void Trainer::learn(int epochs, int batch_size) {
    Loader loader(data_path, batch_size);

    for (int epoch = 0; epoch < epochs; epoch++) {
        loader.reset();

        // Train
        int n_batch = 0;
        double avg_train_loss = 0;
        while (true) {
            try {
                Batch b = loader.get_train_batch();
                double loss = 0;
                train_step(b, &loss);
                n_batch++;
                avg_train_loss = avg_train_loss + (1.0 / n_batch) * (loss - avg_train_loss);

                std::cout << "\rEpoch: " << epoch << " Batch: " << n_batch << "/"
                          << loader.get_n_train_batches() << " Train Loss: " << avg_train_loss
                          << " ====================================" << std::flush;
            } catch (BatchesExhaustedException& e) {
                break;
            }
        }
        std::cout << "\n";

        fs::path cp(checkpoint_path);
        torch::save(model, (cp / "model.pth").string());
        torch::save(*optimizer, (cp / "optim.pth").string());

        // Test
        n_batch = 0;
        double avg_test_loss = 0;
        while (true) {
            try {
                Batch b = loader.get_test_batch();
                double loss = 0;
                test_step(b, &loss);
                n_batch++;
                avg_test_loss = avg_test_loss + (1.0 / n_batch) * (loss - avg_test_loss);

                std::cout << "\rEpoch: " << epoch << " Batch: " << n_batch << "/"
                          << loader.get_n_test_batches() << " Test Loss: " << avg_test_loss
                          << " ====================================" << std::flush;

            } catch (BatchesExhaustedException& e) {
                break;
            }
        }
        std::cout << "\n";
    }
}

}  // namespace ddpm

#endif