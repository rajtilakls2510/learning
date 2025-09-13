#ifndef TRAINER_CPP
#define TRAINER_CPP

#include "trainer.h"

namespace trainer {
    namespace fs = std::filesystem;
    using namespace MNIST;
    
    Trainer::Trainer(std::string data_path, std::string checkpoint_path, bool use_cpu) : data_path(data_path), checkpoint_path(checkpoint_path) {
        fs::path cp(checkpoint_path);
        if(!fs::exists(cp)) {
            std::cout << cp << " does not exist. Creating...\n";
            fs::create_directories(cp);    
        }
        if (!use_cpu) 
            device = torch::cuda::is_available() ? torch::Device(torch::kCUDA) 
                                                : torch::Device(torch::kCPU);
        
        std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";
        
        model = vit::ViT(
            /* img_size */ 28, 
            /* patch_size */ 4, 
            /* num_classes */ 10, 
            /* embed_dim */ 256, 
            /* depth */ 8, 
            /* heads */ 4, 
            /* mlp_dim */ 1024, 
            /* n_channels */ 1);
        optimizer = std::make_shared<torch::optim::Adam>(model->parameters(), torch::optim::AdamOptions(1e-4)); // Learning rate: 5e-4            
        if (!fs::exists(cp/"model.pth")) {
            std::cout << "Did not find model. Saving initial model...\n";
            torch::save(model, (cp / "model.pth").string());
            torch::save(*optimizer, (cp / "optim.pth").string());
        } else {
            torch::load(model, (cp / "model.pth").string());
            torch::load(*optimizer, (cp / "optim.pth").string());
            std::cout << "Loaded model and optimizer.\n";
        }
        model->to(device);
    }

    void Trainer::train_step(Batch batch, double* loss, double* accuracy) {
        model->train();
        auto images = batch.images.to(device);
        auto labels = batch.labels.to(device);
        labels = labels.to(torch::kLong);

        // std::cout << "labels_size: " << labels.sizes() << "\n";

        auto outputs = model->forward(images);

        // std::cout << "outputs_size: " << outputs.sizes() << "\n";
        torch::nn::CrossEntropyLoss criterion;
        auto loss_tensor = criterion(outputs, labels);

        optimizer->zero_grad();
        loss_tensor.backward();
        optimizer->step();

        auto preds = outputs.argmax(1);
        auto correct = preds.eq(labels).sum().item<int64_t>();
        double acc = static_cast<double>(correct) / labels.size(0);

        *loss = loss_tensor.item<double>();
        *accuracy = acc;
    }

    void Trainer::test_step(Batch batch, double* loss, double* accuracy) {
        model->eval();  
        auto images = batch.images.to(device);
        auto labels = batch.labels.to(device);


        labels = labels.to(torch::kLong);

        torch::NoGradGuard no_grad;

        auto outputs = model->forward(images);


        torch::nn::CrossEntropyLoss criterion;
        auto loss_tensor = criterion(outputs, labels);

        auto preds = outputs.argmax(1);
        auto correct = preds.eq(labels).sum().item<int64_t>();
        double acc = static_cast<double>(correct) / labels.size(0);

        *loss = loss_tensor.item<double>();
        *accuracy = acc;
    }

    void Trainer::learn(int epochs, int batch_size) {

        Loader loader(data_path, batch_size);

        for (int epoch=0; epoch < epochs; epoch++) {
            loader.reset();

            // Train
            int n_batch = 0;
            double avg_train_loss = 0, avg_train_acc = 0;
            while(true) {
                try{
                    Batch b = loader.get_train_batch();
                    double loss=0, acc=0;
                    train_step(b, &loss, &acc);
                    n_batch++;
                    avg_train_loss = avg_train_loss + (1.0 / n_batch) * (loss - avg_train_loss);
                    avg_train_acc = avg_train_acc + (1.0 / n_batch) * (acc - avg_train_acc);
                    
                    std::cout << "\rEpoch: " << epoch << " Batch: " << n_batch << "/" << loader.get_n_train_batches() << " Train Loss: " << avg_train_loss << " Train Acc: " << avg_train_acc << " ====================================" << std::flush;
                } catch (BatchesExhaustedException& e) { break; }
            }
            std::cout << "\n";

            // Test
            n_batch = 0;
            double avg_test_loss = 0, avg_test_acc = 0;
            while(true) {
                try{
                    Batch b = loader.get_test_batch();
                    double loss=0, acc=0;
                    test_step(b, &loss, &acc);
                    n_batch++;
                    avg_test_loss = avg_test_loss + (1.0 / n_batch) * (loss - avg_test_loss);
                    avg_test_acc = avg_test_acc + (1.0 / n_batch) * (acc - avg_test_acc);
                    
                    std::cout << "\rEpoch: " << epoch << " Batch: " << n_batch << "/" << loader.get_n_test_batches() << " Test Loss: " << avg_test_loss << " Test Acc: " << avg_test_acc << " ====================================" << std::flush;
                    
                } catch(BatchesExhaustedException& e) { break; }
            }
            std::cout << "\n";
            
            fs::path cp(checkpoint_path);
            torch::save(model, (cp / "model.pth").string());
            torch::save(*optimizer, (cp / "optim.pth").string());

        }
    }


}

#endif