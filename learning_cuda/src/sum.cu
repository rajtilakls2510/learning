#include <cuda_runtime.h>

#include <chrono>
#include <cuda/atomic>
#include <iostream>
#include <random>
#include <thread>
#define THREADS_PER_BLOCK 256

using namespace std::chrono;

void initArray(double *arr, const size_t kNumElements)
{
    for (size_t i = 0; i < kNumElements; i++)
    {
        arr[i] = (std::rand() % 100) + 1;
    }
}

void printArray(double *arr, const size_t kNumElements)
{
    std::cout << "\n[";
    for (size_t i = 0; i < kNumElements; i++)
    {
        std::cout << arr[i] << ", ";
    }
    std::cout << "]\n";
}

__global__ void vecSumK(const double *A, const double *B, double *C, const size_t kNumElements)
{
    int ti = blockDim.x * blockIdx.x + threadIdx.x;
    if (ti >= kNumElements)
        return;
    C[ti] = A[ti] + B[ti];
}

int ceil_div(size_t N) { return (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; }

__global__ void warmup() { return; }

bool checkSum(const double *A, const double *B, const double *C, const size_t kNumElements)
{
    for (size_t i = 0; i < kNumElements; i++)
    {
        if (abs(C[i] - (A[i] + B[i])) > 1e9)
            return false;
    }
    return true;
}

void sum(const size_t kNumElements)
{
    double *A = nullptr;
    double *B = nullptr;
    double *C = nullptr;

    warmup<<<1, 1>>>();
    cudaDeviceSynchronize();

    auto t1 = high_resolution_clock::now();

    cudaMallocManaged(&A, kNumElements * sizeof(double));
    cudaMallocManaged(&B, kNumElements * sizeof(double));
    cudaMallocManaged(&C, kNumElements * sizeof(double));

    auto t2 = high_resolution_clock::now();
    std::cout << "Malloc time: " << duration_cast<microseconds>(t2 - t1).count() << " us\n";

    initArray(A, kNumElements);
    initArray(B, kNumElements);
    cudaMemset(C, 0, kNumElements * sizeof(double));
    // int device;
    // cudaGetDevice(&device);

    // cudaMemPrefetchAsync(A, kNumElements * sizeof(double), device);
    // cudaMemPrefetchAsync(B, kNumElements * sizeof(double), device);
    // cudaMemPrefetchAsync(C, kNumElements * sizeof(double), device);

    // cudaDeviceSynchronize();

    auto t3 = high_resolution_clock::now();
    std::cout << "Init time: " << duration_cast<microseconds>(t3 - t2).count() << " us\n";

    int bpg = ceil_div(kNumElements);
    vecSumK<<<bpg, THREADS_PER_BLOCK>>>(A, B, C, kNumElements);
    cudaDeviceSynchronize();
    auto t4 = high_resolution_clock::now();
    std::cout << "Sum time: " << duration_cast<microseconds>(t4 - t3).count() << " us\n";

    bool match = checkSum(A, B, C, kNumElements);
    auto t5 = high_resolution_clock::now();
    std::cout << "Match time: " << duration_cast<microseconds>(t5 - t4).count() << " us\n";

    std::cout << "Match: " << std::boolalpha << match << "\n";
}

int main(int argc, char *argv[])
{
    sum(1 << 27);
    return 0;
}