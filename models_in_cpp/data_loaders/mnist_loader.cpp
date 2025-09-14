#ifndef MNIST_LOADER_CPP
#define MNIST_LOADER_CPP

#include "mnist_loader.hpp"

using namespace MNIST;

Loader::Loader(std::string data_path, int batch_size)
    : gen(std::random_device{}()), batch_size(batch_size) {
    train_images_path = data_path + "train-images.idx3-ubyte";
    train_labels_path = data_path + "train-labels.idx1-ubyte";
    test_images_path = data_path + "t10k-images.idx3-ubyte";
    test_labels_path = data_path + "t10k-labels.idx1-ubyte";
    std::tie(train_images, train_labels) = read_images_labels(train_images_path, train_labels_path);
    std::tie(test_images, test_labels) = read_images_labels(test_images_path, test_labels_path);
    n_train_batches = train_images.size(0) / batch_size;
    n_test_batches = test_images.size(0) / batch_size;
}

void Loader::reset() {
    c_train_batch = 0;
    c_test_batch = 0;

    // Shuffle training data
    auto train_perm = torch::randperm(train_images.size(0), torch::kLong);
    train_images = train_images.index_select(0, train_perm);
    train_labels = train_labels.index_select(0, train_perm);

    // Shuffle testing data
    auto test_perm = torch::randperm(test_images.size(0), torch::kLong);
    test_images = test_images.index_select(0, test_perm);
    test_labels = test_labels.index_select(0, test_perm);
}

std::pair<torch::Tensor, torch::Tensor> Loader::read_images_labels(
        std::string& images_path, std::string& labels_path) {
    std::ifstream labels_file(labels_path, std::ios::binary);
    std::ifstream images_file(images_path, std::ios::binary);
    if (!labels_file.is_open() || !images_file.is_open()) {
        throw std::runtime_error("Failed to open MNIST file(s).");
    }

    // Read label file header
    uint32_t magic = 0, num_items = 0;
    labels_file.read(reinterpret_cast<char*>(&magic), 4);
    labels_file.read(reinterpret_cast<char*>(&num_items), 4);
    magic = __builtin_bswap32(magic);  // Convert from big-endian
    num_items = __builtin_bswap32(num_items);
    if (magic != 2049) throw std::runtime_error("Invalid label file magic number!");

    std::vector<uint8_t> labels(num_items);
    labels_file.read(reinterpret_cast<char*>(labels.data()), num_items);
    torch::Tensor labels_tensor =
            torch::from_blob(labels.data(), {static_cast<long>(num_items)}, torch::kUInt8).clone();

    // Read image file header
    uint32_t num_images = 0, rows = 0, cols = 0;
    images_file.read(reinterpret_cast<char*>(&magic), 4);
    images_file.read(reinterpret_cast<char*>(&num_images), 4);
    images_file.read(reinterpret_cast<char*>(&rows), 4);
    images_file.read(reinterpret_cast<char*>(&cols), 4);
    magic = __builtin_bswap32(magic);
    num_images = __builtin_bswap32(num_images);
    rows = __builtin_bswap32(rows);
    cols = __builtin_bswap32(cols);

    if (magic != 2051) throw std::runtime_error("Invalid image file magic number!");

    std::vector<uint8_t> images(num_images * rows * cols);
    images_file.read(reinterpret_cast<char*>(images.data()), images.size());

    torch::Tensor images_tensor = torch::from_blob(
                                          images.data(),
                                          {static_cast<long>(num_images),
                                           1,
                                           static_cast<long>(rows),
                                           static_cast<long>(cols)},
                                          torch::kUInt8)
                                          .clone();

    return {images_tensor.to(torch::kFloat32).div(255.0), labels_tensor.to(torch::kLong)};
}

torch::Tensor Loader::get_sample_indices(int c_batch) {
    std::vector<int64_t> sample_indices(batch_size);
    for (int i = 0; i < batch_size; i++) sample_indices[i] = c_batch * batch_size + i;
    return torch::tensor(sample_indices, torch::kLong);
}

Batch Loader::get_train_batch() {
    if (c_train_batch < n_train_batches) {
        auto idx_tensor = get_sample_indices(c_train_batch);
        torch::Tensor images_batch = train_images.index_select(0, idx_tensor);
        torch::Tensor labels_batch = train_labels.index_select(0, idx_tensor);
        c_train_batch++;
        return Batch(images_batch, labels_batch);
    } else {
        throw BatchesExhaustedException("Training batches exhaused");
    }
}

Batch Loader::get_test_batch() {
    if (c_test_batch < n_test_batches) {
        auto idx_tensor = get_sample_indices(c_test_batch);
        torch::Tensor images_batch = test_images.index_select(0, idx_tensor);
        torch::Tensor labels_batch = test_labels.index_select(0, idx_tensor);
        c_test_batch++;
        return Batch(images_batch, labels_batch);
    } else {
        throw BatchesExhaustedException("Testing batches exhaused");
    }
}

// ======================= Visualization and verification code (TO BE REMOVED LATER)
// ================

cv::Mat Loader::tensor_to_cv_mat(torch::Tensor image_tensor) {
    // Step 1: Convert from float [0, 1] to uint8 [0, 255]
    image_tensor = image_tensor.mul(255).clamp(0, 255).to(torch::kU8);

    // Step 2: Ensure tensor is on CPU and contiguous
    image_tensor = image_tensor.cpu().contiguous();

    // Step 3: Convert CHW → HWC if 3-channel
    if (image_tensor.sizes().size() == 3 && image_tensor.size(0) == 3) {
        image_tensor = image_tensor.permute({1, 2, 0});  // CHW to HWC
        return cv::Mat(
                image_tensor.size(0), image_tensor.size(1), CV_8UC3, image_tensor.data_ptr());
    }
    // Step 4: If grayscale [1, H, W], convert to [H, W]
    else if (image_tensor.sizes().size() == 3 && image_tensor.size(0) == 1) {
        image_tensor = image_tensor.squeeze();  // remove channel dim -> [H, W]
        return cv::Mat(
                image_tensor.size(0), image_tensor.size(1), CV_8UC1, image_tensor.data_ptr());
    } else if (image_tensor.sizes().size() == 2) {  // Already [H, W]
        return cv::Mat(
                image_tensor.size(0), image_tensor.size(1), CV_8UC1, image_tensor.data_ptr());
    } else {
        throw std::runtime_error("Unsupported tensor shape for image conversion");
    }
}

#endif