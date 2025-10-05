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

    // std::vector<int> cm = {2, 4, 4};
    // model = unet::UNetModel(
    //         /*img size*/ 28,
    //         /*in channels*/ 1,
    //         /*model channels*/ 64,
    //         /*out channels*/ 1,
    //         /*num res blocks*/ 1,
    //         /*dropout*/ 0.1,
    //         /*num heads*/ 4,
    //         /*begin attention after level*/ 0,
    //         /*channel multipliers*/ cm);
    model = unet::SimpleUNet(
            /*img_size*/ 28,
            /*in_channels*/ 1,
            /*out_channels*/ 2,
            /*time_dim*/ 256,
            /*channel_dims*/ std::vector<int>{64, 256, 256});
    optimizer = std::make_shared<torch::optim::Adam>(
            model->parameters(), torch::optim::AdamOptions(1e-3));
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
    posterior_variance = ((1 - alphas_cumprod_prev) * betas) / (1.0 - alphas_cumprod);
    posterior_log_var = torch::log(torch::cat(
            {posterior_variance.index({Slice(1, 2)}), posterior_variance.index({Slice(1, None)})}));
    posterior_mean_coef1 = ((betas * torch::sqrt(alphas_cumprod_prev)) / (1.0 - alphas_cumprod));
    posterior_mean_coef2 =
            (torch::sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod));
    sqrt_recip_alphas_cumprod = torch::sqrt(1.0 / alphas_cumprod);
    sqrt_recipm1_alphas_cumprod = torch::sqrt(1.0 / alphas_cumprod - 1.0);
}

torch::Tensor Trainer::q_sample(torch::Tensor x_start, torch::Tensor t, torch::Tensor noise) {
    return extract(sqrt_alphas_cumprod, t, x_start.sizes()) * x_start +
           extract(sqrt_one_minus_alphas_cumprod, t, x_start.sizes()) * noise;
}

torch::Tensor Trainer::predict_xstart_from_eps(
        torch::Tensor x_t, torch::Tensor t, torch::Tensor noise) {
    return (extract(sqrt_recip_alphas_cumprod, t, x_t.sizes()) * x_t +
            extract(sqrt_recipm1_alphas_cumprod, t, x_t.sizes()) * noise);
}

torch::Tensor Trainer::normal_kl(
        torch::Tensor mean1, torch::Tensor logvar1, torch::Tensor mean2, torch::Tensor logvar2) {
    torch::Tensor exp_log_vars = torch::exp(logvar1 - logvar2);
    torch::Tensor mean_diff_sq = torch::pow(mean1 - mean2, 2);
    return 0.5 * (-1.0 + logvar2 - logvar1 + exp_log_vars + mean_diff_sq * torch::exp(-logvar2));
}

void Trainer::train_step(Batch batch, double* loss) {
    model->train();
    auto images = batch.images.to(device);
    images = 2 * images - 1;  // Scale between [-1,1]
    torch::Tensor t, noise, x_noisy, true_mean, true_log_var;
    {
        torch::NoGradGuard no_grad;
        t = torch::randint(1, max_diffusion_time + 1, {/*batch*/ batch.batch_size}, torch::kLong)
                    .to(device);
        noise = torch::randn_like(images).to(device);
        x_noisy = q_sample(images, t - 1, noise);

        true_mean = extract(posterior_mean_coef1, t - 1, noise.sizes()) * images +
                    extract(posterior_mean_coef2, t - 1, noise.sizes()) * x_noisy;
        true_log_var = extract(posterior_log_var, t - 1, noise.sizes());
    }

    auto outputs = model->forward(x_noisy, t);
    torch::Tensor noise_predicted = outputs.index({Slice(), Slice(0, 1)});   // outputs[:, 0:1]
    torch::Tensor var_predicted = outputs.index({Slice(), Slice(1, None)});  // outputs[:, 1:]
    var_predicted.clamp_(-1.0, 1.0);

    torch::Tensor pred_xstart =
            predict_xstart_from_eps(x_noisy, t - 1, noise_predicted).clamp(-1.0, 1.0);
    torch::Tensor mean_predicted =
            extract(posterior_mean_coef1, t - 1, x_noisy.sizes()) * pred_xstart +
            extract(posterior_mean_coef2, t - 1, noise.sizes()) * x_noisy;

    torch::Tensor min_log = extract(posterior_log_var, t - 1, x_noisy.sizes());
    torch::Tensor max_log = extract(torch::log(betas), t - 1, x_noisy.sizes());
    torch::Tensor frac = (var_predicted + 1) / 2;
    torch::Tensor log_var_predicted = frac * max_log + (1 - frac) * min_log;

    torch::Tensor kl = normal_kl(true_mean, true_log_var, mean_predicted, log_var_predicted);
    torch::Tensor loss_tensor = torch::mean(kl) / torch::log(torch::tensor(2.0f));

    optimizer->zero_grad();
    loss_tensor.backward();
    optimizer->step();

    *loss = loss_tensor.item<double>();
}

void Trainer::test_step(Batch batch, double* loss) {
    model->eval();
    auto images = batch.images.to(device);
    images = 2 * images - 1;  // Scale between [-1,1]
    torch::Tensor t, noise, x_noisy, true_mean, true_log_var;

    torch::NoGradGuard no_grad;
    t = torch::randint(1, max_diffusion_time + 1, {/*batch*/ batch.batch_size}, torch::kLong)
                .to(device);
    noise = torch::randn_like(images).to(device);
    x_noisy = q_sample(images, t - 1, noise);
    true_mean = extract(posterior_mean_coef1, t - 1, noise.sizes()) * images +
                extract(posterior_mean_coef2, t - 1, noise.sizes()) * x_noisy;
    true_log_var = extract(posterior_log_var, t - 1, noise.sizes());

    auto outputs = model->forward(x_noisy, t);
    torch::Tensor noise_predicted = outputs.index({Slice(), Slice(0, 1)});   // outputs[:, 0:1]
    torch::Tensor var_predicted = outputs.index({Slice(), Slice(1, None)});  // outputs[:, 1:]
    var_predicted.clamp_(-1.0, 1.0);

    torch::Tensor pred_xstart =
            predict_xstart_from_eps(x_noisy, t - 1, noise_predicted).clamp(-1.0, 1.0);
    torch::Tensor mean_predicted =
            extract(posterior_mean_coef1, t - 1, x_noisy.sizes()) * pred_xstart +
            extract(posterior_mean_coef2, t - 1, noise.sizes()) * x_noisy;

    torch::Tensor min_log = extract(posterior_log_var, t - 1, x_noisy.sizes());
    torch::Tensor max_log = extract(torch::log(betas), t - 1, x_noisy.sizes());
    torch::Tensor frac = (var_predicted + 1) / 2;
    torch::Tensor log_var_predicted = frac * max_log + (1 - frac) * min_log;

    torch::Tensor kl = normal_kl(true_mean, true_log_var, mean_predicted, log_var_predicted);
    torch::Tensor loss_tensor = torch::mean(kl) / torch::log(torch::tensor(2.0f));

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