#ifndef NET_UTILS_H
#define NET_UTILS_H

#include <torch/torch.h>
#include <string>

namespace net {

    inline torch::Device getDevice(std::string device) {
        return device == "cuda"? torch::kCUDA : torch::kCPU;
    }

    inline std::string get_size(torch::Tensor x){
        std::string output_str = "Tensor[";
        for(auto& size: x.sizes())
            output_str += std::to_string(size) + ", ";
        output_str += "]";
        return output_str;
    }

    template <typename T>
    int64_t count_parameters(const torch::nn::ModuleHolder<T>& model) {
        int64_t total_params = 0;
        // Iterate over all the parameters in the model
        for (const auto& param : model->parameters()) {
            total_params += param.numel(); // numel() gives the total number of elements in the tensor
        }
        return total_params;
    }

}   // namespace vit


#endif