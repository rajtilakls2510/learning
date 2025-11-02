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
            /* n_res_layers */ 3,
            /* n_embed */ 128,
            /* gamma_min */ -13.3,
            /* gamma_max */ 5.0,
            max_diffusion_time);
    gamma = unet::NoiseNet(
            /* mid_features */ 1024,
            /* gamma_min */ -13.3,
            /* gamma_max*/ 5.0);
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
    torch::Tensor x_vals = torch::arange(0, vocab_size + 1).unsqueeze(-1) / vocab_size;
    x_vals = x_vals.repeat({1, 1}).to(z.device());
    x_vals = encode(x_vals).transpose(1, 0).unsqueeze(0).unsqueeze(-2).unsqueeze(-2);
    torch::Tensor inv_stddev = torch::exp(-0.5 * g_0);
    torch::Tensor logits = -0.5 * torch::pow((z.unsqueeze(-1) - x_vals) / inv_stddev, 2);
    return torch::nn::functional::log_softmax(logits, /*dim*/ -1);
}

torch::Tensor Trainer::logprob(torch::Tensor x, torch::Tensor z, torch::Tensor g_0) {
    x = (x * vocab_size).round().to(torch::kLong);
    auto x_onehot =
            torch::nn::functional::one_hot(x, vocab_size + 1).to(torch::kFloat).to(x.device());
    auto logprobs = decode(z, g_0);
    return (x_onehot * logprobs).mean({1, 2, 3, 4});
}

void Trainer::train_step(Batch batch, double* model_loss, double* classifier_loss) {
    model->train();
    gamma->train();
    // classifier->train();
    auto x = batch.images.to(device);
    auto f = encode(x);
    int b = x.size(0);
    auto t_0 = torch::zeros({1, 1}).to(device);
    auto g_0 = gamma(t_0);
    auto t_1 = torch::ones({1, 1}).to(device);
    auto g_1 = gamma(t_1);

    auto var_0 = torch::sigmoid(g_0);
    auto var_1 = torch::sigmoid(g_1);

    // Reconstruction Loss
    torch::Tensor eps_0 = torch::rand_like(x).to(device);
    auto z_0 = torch::sqrt(1.0 - var_0) * f + torch::sqrt(var_0) * eps_0;
    auto z_0_rescaled = f + torch::exp(0.5 * g_0) * eps_0;
    auto loss_recon = -logprob(x, z_0_rescaled, g_0);

    // Latent Loss
    auto mean1_sq = (1.0 - var_1) * torch::pow(f, 2);
    auto loss_klz = 0.5 * (mean1_sq + var_1 - torch::log(var_1) - 1.0f).mean({1, 2, 3});

    // Diffusion Loss

    // antithetic time sampling
    float t0 = torch::rand(1).item<float>();
    auto offsets = torch::arange(0.0, 1.0, 1.0 / b).to(device);
    auto t = torch::fmod(t0 + offsets, 1.0).unsqueeze(-1);
    t = torch::ceil(t * max_diffusion_time) / max_diffusion_time;

    // sample z_t
    auto g_t = gamma(t);
    auto var_t = torch::sigmoid(g_t).reshape({b, 1, 1, 1});
    auto eps = torch::rand_like(f);
    auto z_t = torch::sqrt(1.0f - var_t) * f + torch::sqrt(var_t) * eps;
    auto eps_hat = model(z_t, g_t.squeeze(-1));
    auto loss_diff_mse = torch::pow(eps - eps_hat, 2).mean({1, 2, 3});

    auto s = t - (1.0 / max_diffusion_time);
    auto g_s = gamma(s);
    auto loss_diff = 0.5 * max_diffusion_time * torch::expm1(g_t - g_s) * loss_diff_mse;

    auto loss_tensor = (loss_recon + loss_klz + loss_diff).mean();

    optimizer->zero_grad();
    gamma_optimizer->zero_grad();
    loss_tensor.backward();
    optimizer->step();
    gamma_optimizer->step();

    *model_loss = loss_tensor.item<float>();
    *classifier_loss = 0.0;
}

void Trainer::test_step(Batch batch, double* model_loss, double* classifier_loss) {
    model->eval();
    gamma->eval();
    // classifier->eval();

    torch::NoGradGuard no_grad;
    auto x = batch.images.to(device);
    auto f = encode(x);
    int b = x.size(0);
    auto t_0 = torch::zeros({1, 1}).to(device);
    auto g_0 = gamma(t_0);
    auto t_1 = torch::ones({1, 1}).to(device);
    auto g_1 = gamma(t_1);

    auto var_0 = torch::sigmoid(g_0);
    auto var_1 = torch::sigmoid(g_1);

    // Reconstruction Loss
    torch::Tensor eps_0 = torch::rand_like(x).to(device);
    auto z_0 = torch::sqrt(1.0 - var_0) * f + torch::sqrt(var_0) * eps_0;
    auto z_0_rescaled = f + torch::exp(0.5 * g_0) * eps_0;
    auto loss_recon = -logprob(x, z_0_rescaled, g_0);

    // Latent Loss
    auto mean1_sq = (1.0 - var_1) * torch::pow(f, 2);
    auto loss_klz = 0.5 * (mean1_sq + var_1 - torch::log(var_1) - 1.0f).mean({1, 2, 3});

    // Diffusion Loss

    // antithetic time sampling
    float t0 = torch::rand(1).item<float>();
    auto offsets = torch::arange(0.0, 1.0, 1.0 / b).to(device);
    auto t = torch::fmod(t0 + offsets, 1.0).unsqueeze(-1);
    t = torch::ceil(t * max_diffusion_time) / max_diffusion_time;

    // sample z_t
    auto g_t = gamma(t);
    auto var_t = torch::sigmoid(g_t).reshape({b, 1, 1, 1});
    auto eps = torch::rand_like(f);
    auto z_t = torch::sqrt(1.0f - var_t) * f + torch::sqrt(var_t) * eps;
    auto eps_hat = model(z_t, g_t.squeeze(-1));
    auto loss_diff_mse = torch::pow(eps - eps_hat, 2).mean({1, 2, 3});

    auto s = t - (1.0 / max_diffusion_time);
    auto g_s = gamma(s);
    auto loss_diff = 0.5 * max_diffusion_time * torch::expm1(g_t - g_s) * loss_diff_mse;

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