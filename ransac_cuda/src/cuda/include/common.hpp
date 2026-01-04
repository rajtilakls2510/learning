#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <cuda_runtime.h>
#include <stdio.h>

namespace ransac {

#define CUDA_CHECK(ans)                       \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDAassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

}  // namespace ransac

#endif