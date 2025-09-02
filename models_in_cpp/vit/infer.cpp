#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "vit.h"      
#include "data_loader.h"

using namespace MNIST;

// Function to display MNIST image tensor
void show_image(torch::Tensor img, const std::string &window_name) {
    // MNIST image is 1x28x28 (CHW)
    img = img.squeeze().detach().cpu();
    img = img.mul(255).clamp(0, 255).to(torch::kU8);

    cv::Mat mat(cv::Size(28, 28), CV_8U, img.data_ptr());
    cv::resize(mat, mat, cv::Size(280, 280), 0, 0, cv::INTER_NEAREST);

    cv::imshow(window_name, mat);
    cv::waitKey(2000);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: ./infer <data_path> <checkpoint_path>\n";
        return -1;
    }

    std::string data_path = argv[1];
    std::string checkpoint_path = argv[2];

    // Load model
    auto model = vit::ViT(28, 4, 10, 256, 8, 4, 1024, 1);
    torch::load(model, checkpoint_path + "/model.pth");
    model->to(torch::kCUDA);
    model->eval();

    // Data loader
    Loader loader(data_path, 1); // batch size = 1
    loader.reset();

    torch::NoGradGuard no_grad;
    for (int i = 0; i < 100; ++i) {  // Show first 10 images
        Batch b = loader.get_test_batch();
        auto imgs = b.images.to(torch::kCUDA);  
        auto logits = model->forward(imgs);
        auto pred = logits.argmax(1);

        std::cout << "Predicted: " << pred[0].item<int>() << " | Actual: " << b.labels[0].item<int>() << "\n";
        show_image(b.images[0], "img");
    }

    return 0;
}
