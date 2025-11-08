#ifndef TRAINER_CPP
#define TRAINER_CPP

#include "trainer.h"

namespace vdm {
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

    model = unet::ScoreModel(
            /* in_out_channels */ 1,
            /* n_res_layers */ 5,
            /* n_embed */ 128,
            max_diffusion_time);
    gamma = unet::NoiseNet(
            /* mid_features */ 1024,
            /* gamma_min */ gamma_min,
            /* gamma_max*/ gamma_max);
    optimizer = std::make_shared<torch::optim::Adam>(
            model->parameters(), torch::optim::AdamOptions(3e-4));
    gamma_optimizer = std::make_shared<torch::optim::Adam>(
            gamma->parameters(), torch::optim::AdamOptions(3e-4));
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

    if (!fs::exists(cp / "gamma.pth")) {
        std::cout << "Did not find learnable schedule. Saving initial learnable schedule...\n";
        torch::save(gamma, (cp / "gamma.pth").string());
        torch::save(*gamma_optimizer, (cp / "gamma_optim.pth").string());
    } else {
        torch::load(gamma, (cp / "gamma.pth").string());
        torch::load(*gamma_optimizer, (cp / "gamma_optim.pth").string());
        std::cout << "Loaded learnable schedule and optimizer.\n";
    }
    gamma->to(device);

    std::cout << "Num Parameters (model): " << net::count_parameters(model) << "\n";
    std::cout << "Num Parameters (learnable schedule): " << net::count_parameters(gamma) << "\n";
}

torch::Tensor Trainer::encode(torch::Tensor x) { return 2 * x - 1; }

torch::Tensor Trainer::decode(torch::Tensor z, torch::Tensor g_0) {
    // z: [B,C,H,W], g_0: [1,1]
    torch::Tensor x_vals = torch::arange(0, vocab_size + 1).unsqueeze(-1) / vocab_size;  // [VS+1,1]
    x_vals = x_vals.repeat({1, 1}).to(z.device());                                       // [VS+1,1]
    x_vals = encode(x_vals).transpose(1, 0).unsqueeze(0).unsqueeze(-2).unsqueeze(
            -2);                                        // [1,1,1,1,VS+1]
    torch::Tensor inv_stddev = torch::exp(-0.5 * g_0);  // [1,1]
    torch::Tensor logits =
            -0.5 * torch::pow((z.unsqueeze(-1) - x_vals) / inv_stddev, 2);  // [B,C,H,W,VS+1]
    return torch::nn::functional::log_softmax(logits, /*dim*/ -1);          // [B,C,H,W,VS+1]
}

torch::Tensor Trainer::logprob(torch::Tensor x, torch::Tensor z, torch::Tensor g_0) {
    // x: [B,C,H,W], z: [B,C,H,W], g_0: [1,1]
    x = (x * vocab_size).round().to(torch::kLong);  // [B,C,H,W]
    auto x_onehot = torch::nn::functional::one_hot(x, vocab_size + 1)
                            .to(torch::kFloat)
                            .to(x.device());          // [B,C,H,W,VS+1]
    auto logprobs = decode(z, g_0);                   // [B,C,H,W,VS+1]
    return (x_onehot * logprobs).mean({1, 2, 3, 4});  // [B,]
}

void Trainer::train_step(Batch batch, double* model_loss, double* classifier_loss) {
    model->train();
    gamma->train();
    // classifier->train();
    auto x = batch.images.to(device);  // [B,C,H,W]
    auto f = encode(x);                // [B,C,H,W]
    int b = x.size(0);
    auto t_0 = torch::zeros({1, 1}).to(device);  // [1,1]
    auto g_0 = gamma(t_0);                       // [1,1]
    auto t_1 = torch::ones({1, 1}).to(device);   // [1,1]
    auto g_1 = gamma(t_1);                       // [1,1]

    auto var_0 = torch::sigmoid(g_0);  // [1,1]
    auto var_1 = torch::sigmoid(g_1);  // [1,1]

    // Reconstruction Loss
    torch::Tensor eps_0;
    {
        torch::NoGradGuard no_grad;
        eps_0 = torch::randn_like(x).to(device);  // [B,C,H,W]
    }
    auto z_0 = torch::sqrt(1.0 - var_0) * f + torch::sqrt(var_0) * eps_0;  // [B,C,H,W]
    auto z_0_rescaled = f + torch::exp(0.5 * g_0) * eps_0;                 // [B,C,H,W]
    auto loss_recon = -logprob(x, z_0_rescaled, g_0);                      // [B,]

    // Latent Loss
    auto mean1_sq = (1.0 - var_1) * torch::pow(f, 2);  // [B,C,H,W]
    auto loss_klz = 0.5 * (mean1_sq + var_1 - torch::log(var_1) - 1.0f).mean({1, 2, 3});  // [B,]

    // Diffusion Loss

    // antithetic time sampling
    float t0 = torch::rand(1).item<float>();
    auto offsets = torch::arange(0.0, 1.0, 1.0 / b).to(device);    // [B,]
    auto t = torch::fmod(t0 + offsets, 1.0).unsqueeze(-1);         // [B,1]
    t = torch::ceil(t * max_diffusion_time) / max_diffusion_time;  // [B,1]

    // sample z_t
    auto g_t = gamma(t);                                     // [B,1]
    auto var_t = torch::sigmoid(g_t).reshape({b, 1, 1, 1});  // [B,1,1,1]
    torch::Tensor eps;
    {
        torch::NoGradGuard no_grad;
        eps = torch::randn_like(f);  // [B,C,H,W]
    }
    auto z_t = torch::sqrt(1.0 - var_t) * f + torch::sqrt(var_t) * eps;            // [B,C,H,W]
    auto eps_hat = model(z_t, g_t.squeeze(-1), g_0.squeeze(-1), g_1.squeeze(-1));  // [B,C,H,W]
    auto loss_diff_mse = torch::pow(eps - eps_hat, 2).mean({1, 2, 3});             // [B,]

    auto s = t - (1.0 / max_diffusion_time);  // [B,1]
    auto g_s = gamma(s);                      // [B,1]
    auto loss_diff =
            0.5 * max_diffusion_time * torch::expm1(g_t - g_s).squeeze(-1) * loss_diff_mse;  // [B,]
    auto loss_tensor = (loss_recon + loss_klz + loss_diff).mean();

    optimizer->zero_grad();
    gamma_optimizer->zero_grad();
    loss_tensor.backward();

    // After backward(), before step():
    for (auto& p : gamma->parameters()) {
        if (p.grad().defined() && torch::isnan(p.grad()).any().item<bool>()) {
            std::cout << "NaN in gamma grad!\n";
            break;
        }
    }
    for (auto& p : model->parameters()) {
        if (p.grad().defined() && torch::isnan(p.grad()).any().item<bool>()) {
            std::cout << "NaN in model grad!\n";
            break;
        }
    }

    optimizer->step();
    gamma_optimizer->step();

    // After optimizer step():
    for (auto& p : gamma->parameters()) {
        if (torch::isnan(p).any().item<bool>()) {
            std::cout << "NaN in gamma param after step!\n";
            std::exit(EXIT_FAILURE);
            break;
        }
    }

    *model_loss = loss_tensor.item<float>();
    *classifier_loss = 0.0;
}

void Trainer::test_step(Batch batch, double* model_loss, double* classifier_loss) {
    model->eval();
    gamma->eval();
    // classifier->eval();

    torch::NoGradGuard no_grad;
    auto x = batch.images.to(device);  // [B,C,H,W]
    auto f = encode(x);                // [B,C,H,W]
    int b = x.size(0);
    auto t_0 = torch::zeros({1, 1}).to(device);  // [1,1]
    auto g_0 = gamma(t_0);                       // [1,1]
    auto t_1 = torch::ones({1, 1}).to(device);   // [1,1]
    auto g_1 = gamma(t_1);                       // [1,1]

    auto var_0 = torch::sigmoid(g_0);  // [1,1]
    auto var_1 = torch::sigmoid(g_1);  // [1,1]

    // Reconstruction Loss
    torch::Tensor eps_0 = torch::randn_like(x).to(device);                 // [B,C,H,W]
    auto z_0 = torch::sqrt(1.0 - var_0) * f + torch::sqrt(var_0) * eps_0;  // [B,C,H,W]
    auto z_0_rescaled = f + torch::exp(0.5 * g_0) * eps_0;                 // [B,C,H,W]
    auto loss_recon = -logprob(x, z_0_rescaled, g_0);                      // [B,]

    // Latent Loss
    auto mean1_sq = (1.0 - var_1) * torch::pow(f, 2);  // [B,C,H,W]
    auto loss_klz = 0.5 * (mean1_sq + var_1 - torch::log(var_1) - 1.0f).mean({1, 2, 3});  // [B,]

    // Diffusion Loss

    // antithetic time sampling
    float t0 = torch::rand(1).item<float>();
    auto offsets = torch::arange(0.0, 1.0, 1.0 / b).to(device);    // [B,]
    auto t = torch::fmod(t0 + offsets, 1.0).unsqueeze(-1);         // [B,1]
    t = torch::ceil(t * max_diffusion_time) / max_diffusion_time;  // [B,1]

    // sample z_t
    auto g_t = gamma(t);                                                           // [B,1]
    auto var_t = torch::sigmoid(g_t).reshape({b, 1, 1, 1});                        // [B,1,1,1]
    torch::Tensor eps = torch::randn_like(f);                                      // [B,C,H,W]
    auto z_t = torch::sqrt(1.0 - var_t) * f + torch::sqrt(var_t) * eps;            // [B,C,H,W]
    auto eps_hat = model(z_t, g_t.squeeze(-1), g_0.squeeze(-1), g_1.squeeze(-1));  // [B,C,H,W]
    auto loss_diff_mse = torch::pow(eps - eps_hat, 2).mean({1, 2, 3});             // [B,]

    auto s = t - (1.0 / max_diffusion_time);  // [B,1]
    auto g_s = gamma(s);                      // [B,1]
    auto loss_diff =
            0.5 * max_diffusion_time * torch::expm1(g_t - g_s).squeeze(-1) * loss_diff_mse;  // [B,]
    auto loss_tensor = (loss_recon + loss_klz + loss_diff).mean();

    *model_loss = loss_tensor.item<float>();
    *classifier_loss = 0.0;
}

void Trainer::learn(int epochs, int batch_size) {
    Loader loader(data_path, batch_size);

    for (int epoch = 0; epoch < epochs; epoch++) {
        loader.reset();

        // Train
        int n_batch = 0;
        double avg_train_model_loss = 0, avg_train_classifier_loss = 0;
        while (true) {
            try {
                Batch b = loader.get_train_batch();
                double model_loss = 0, classifier_loss = 0;
                train_step(b, &model_loss, &classifier_loss);
                n_batch++;
                avg_train_model_loss = avg_train_model_loss +
                                       (1.0 / n_batch) * (model_loss - avg_train_model_loss);
                avg_train_classifier_loss =
                        avg_train_classifier_loss +
                        (1.0 / n_batch) * (classifier_loss - avg_train_classifier_loss);

                std::cout << "\rEpoch: " << epoch << " Batch: " << n_batch << "/"
                          << loader.get_n_train_batches() << " Train Model Loss: "
                          << avg_train_model_loss
                          //   << " Train Classifier Loss: " << avg_train_classifier_loss
                          << " ====================================" << std::flush;
            } catch (BatchesExhaustedException& e) {
                break;
            }
        }
        std::cout << "\n";

        fs::path cp(checkpoint_path);
        torch::save(model, (cp / "model.pth").string());
        torch::save(*optimizer, (cp / "optim.pth").string());
        torch::save(gamma, (cp / "gamma.pth").string());
        torch::save(*gamma_optimizer, (cp / "gamma_optim.pth").string());

        // Test
        n_batch = 0;
        double avg_test_model_loss = 0, avg_test_classifier_loss = 0;
        while (true) {
            try {
                Batch b = loader.get_test_batch();
                double model_loss = 0, classifier_loss = 0;
                test_step(b, &model_loss, &classifier_loss);
                n_batch++;
                avg_test_model_loss =
                        avg_test_model_loss + (1.0 / n_batch) * (model_loss - avg_test_model_loss);
                avg_test_classifier_loss =
                        avg_test_classifier_loss +
                        (1.0 / n_batch) * (classifier_loss - avg_test_classifier_loss);

                std::cout << "\rEpoch: " << epoch << " Batch: " << n_batch << "/"
                          << loader.get_n_test_batches() << " Test Model Loss: "
                          << avg_test_model_loss
                          //   << " Test Classifier Loss: " << avg_test_classifier_loss
                          << " ====================================" << std::flush;

            } catch (BatchesExhaustedException& e) {
                break;
            }
        }
        std::cout << "\n";
    }
}

}  // namespace vdm

#endif