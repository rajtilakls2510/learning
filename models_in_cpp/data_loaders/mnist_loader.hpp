#ifndef MNIST_LOADER_HPP
#define MNIST_LOADER_HPP

#include <torch/torch.h>

#include <fstream>
#include <opencv2/opencv.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace MNIST {

class BatchesExhaustedException : public std::exception {
private:
    std::string message;

public:
    explicit BatchesExhaustedException(const std::string& msg) : message(msg) {}

    const char* what() const noexcept override { return message.c_str(); }
};

struct Batch {
    torch::Tensor images;

    torch::Tensor labels;
    Batch(torch::Tensor images, torch::Tensor labels) : images(images), labels(labels) {}
};

class Loader {
public:
    Loader(std::string data_path, int batch_size);
    void reset();
    std::pair<torch::Tensor, torch::Tensor> read_images_labels(
            std::string& images_path, std::string& labels_path);
    torch::Tensor get_sample_indices(int c_batch);
    Batch get_train_batch();
    Batch get_test_batch();
    cv::Mat tensor_to_cv_mat(torch::Tensor image_tensor);
    int get_n_train_batches() { return n_train_batches; }
    int get_n_test_batches() { return n_test_batches; }

private:
    std::string train_images_path, train_labels_path, test_images_path, test_labels_path;
    torch::Tensor train_images, train_labels, test_images, test_labels;
    std::mt19937 gen;
    int batch_size, n_train_batches, n_test_batches, c_train_batch, c_test_batch;
};

}  // namespace MNIST

#endif