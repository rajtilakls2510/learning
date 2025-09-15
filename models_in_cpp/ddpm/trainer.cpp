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
            /* embed_dim */ 256,
            /* depth */ 8,
            /* heads */ 4,
            /* mlp_dim */ 1024,
            /* n_channels */ 1,
            /* max diffusion time*/ max_diffusion_time);
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
    betas = cosine_beta_schedule(max_diffusion_time).to(device);

    alphas = 1.0 - betas;
    alphas_cumprod = torch::cumprod(alphas, /*axis*/ 0);

    // std::cout << "alphas cumprod: " << alphas_cumprod << "\n";
    alphas_cumprod_prev =
            torch::cat({torch::ones({1}).to(device), alphas_cumprod.index({Slice(0, -1)})});

    // std::cout << "alphas cumprod prev: " << alphas_cumprod_prev << "\n";
    sqrt_alphas_cumprod = torch::sqrt(alphas_cumprod);

    // std::cout << "sqrt alphas cumprod: " << sqrt_alphas_cumprod << "\n";
    sqrt_one_minus_alphas_cumprod = torch::sqrt(1.0 - alphas_cumprod);
    // std::cout << "sqrt 1m alphas cumprod: " << sqrt_one_minus_alphas_cumprod << "\n";

    // torch::Tensor x_start = torch::zeros({3,1,5,5}).to(device);
    // std::cout << "extracted: " << extract(alphas_cumprod, torch::tensor({1, 500,
    // 999}).to(torch::kLong).to(device), x_start.sizes()); std::cerr << "Done \n";
}

torch::Tensor Trainer::q_sample(torch::Tensor x_start, torch::Tensor t, torch::Tensor noise) {
    return extract(sqrt_alphas_cumprod, t, x_start.sizes()) * x_start +
           extract(sqrt_one_minus_alphas_cumprod, t, x_start.sizes()) * noise;
}

void Trainer::train_step(Batch batch, double* loss /*TODO Metrics*/) {
    model->train();
    auto images = batch.images.to(device);
    images = 2 * images - 1;    // Scale between [-1,1]
    torch::Tensor t, noise, x_noisy;
    {
        torch::NoGradGuard no_grad;
        t = torch::randint(1, max_diffusion_time + 1, {/*batch*/ batch.batch_size}, torch::kLong)
                    .to(device);
        noise = torch::randn_like(images).to(device);
        x_noisy = q_sample(images, t - 1, noise);
    }

    // std::cout << "t: " << net::get_size(t) << "\n";
    // std::cout << "noise: " << net::get_size(noise) << "\n";
    // std::cout << "x_noisy: " << net::get_size(x_noisy) << "\n";

    auto outputs = model->forward(x_noisy, t.unsqueeze(-1));

    // std::cout << "outputs: " << net::get_size(outputs) << " " << outputs.index({0,0}) << "\n";

    torch::nn::MSELoss criterion;
    auto loss_tensor = criterion(outputs, noise);

    optimizer->zero_grad();
    loss_tensor.backward();
    optimizer->step();

    *loss = loss_tensor.item<double>();
}

void Trainer::test_step(Batch batch, double* loss /*TODO Metrics*/) {
    model->eval();
    auto images = batch.images.to(device);
    torch::Tensor t, noise, x_noisy;

    torch::NoGradGuard no_grad;
    t = torch::randint(1, max_diffusion_time + 1, {/*batch*/ batch.batch_size}, torch::kLong)
                .to(device);
    noise = torch::randn_like(images).to(device);
    x_noisy = q_sample(images, t - 1, noise);

    auto outputs = model->forward(x_noisy, t.unsqueeze(-1));

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
                // avg_train_acc = avg_train_acc + (1.0 / n_batch) * (acc - avg_train_acc);

                std::cout << "\rEpoch: " << epoch << " Batch: " << n_batch << "/"
                          << loader.get_n_train_batches() << " Train Loss: "
                          << avg_train_loss
                          //<< " Train Acc: " << avg_train_acc
                          << " ====================================" << std::flush;
            } catch (BatchesExhaustedException& e) {
                break;
            }
        }
        std::cout << "\n";

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
                // avg_test_acc = avg_test_acc + (1.0 / n_batch) * (acc - avg_test_acc);

                std::cout << "\rEpoch: " << epoch << " Batch: " << n_batch << "/"
                          << loader.get_n_test_batches() << " Test Loss: "
                          << avg_test_loss
                          //<< " Test Acc: " << avg_test_acc
                          << " ====================================" << std::flush;

            } catch (BatchesExhaustedException& e) {
                break;
            }
        }
        std::cout << "\n";

        fs::path cp(checkpoint_path);
        torch::save(model, (cp / "model.pth").string());
        torch::save(*optimizer, (cp / "optim.pth").string());
    }
}

}  // namespace ddpm

#endif