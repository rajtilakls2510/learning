#include <cuda_runtime.h>

#include <chrono>
#include <cuda/atomic>
#include <iostream>
#include <random>
#include <thread>
#define THREADS_PER_BLOCK 256

using namespace std::chrono;

__global__ void warmup() { return; }

void initArray(double* arr, const size_t kNumElements) {
    for (size_t i = 0; i < kNumElements; i++) {
        arr[i] = (std::rand() % 100) + 1;
    }
}

void printArray(double* arr, const size_t kNumElements) {
    std::cout << "\n[";
    for (size_t i = 0; i < kNumElements; i++) {
        std::cout << arr[i] << ", ";
    }
    std::cout << "]\n";
}

__global__ void atomicSum(const double* A, const size_t kNumElements, double* result) {
    int ti = blockDim.x * blockIdx.x + threadIdx.x;
    if (ti >= kNumElements) return;
    cuda::atomic_ref<double, cuda::thread_scope_device> result_ref(*result);
    result_ref.fetch_add(A[ti]);
}

__global__ void reduceSum(const double* A, const size_t kNumElements, double* result) {
    int ti = blockDim.x * blockIdx.x + threadIdx.x;
    if (ti >= kNumElements) return;

    extern __shared__ double smem[];

    // Load data into shared memory
    smem[threadIdx.x] = A[ti];
    __syncthreads();  // Sync all threads

    // Logarithmically calculate sum
    for (int i = blockDim.x / 2; i >= 1; i = i / 2) {
        if (threadIdx.x >= i) return;  // Half of the threads don't work

        // Add and copy next half data
        smem[threadIdx.x] = smem[threadIdx.x] + smem[threadIdx.x + i];
        __syncthreads();  // Sync threads
    }

    // If it is the first thread, copy data from the start of shared_memory to result
    if (threadIdx.x == 0) {
        result[blockIdx.x] = smem[0];
    }
}

int ceilDiv(size_t N) { return (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK; }

double cpuSum(const double* A, const size_t N) {
    double S = 0;
    for (size_t i = 0; i < N; i++) S += A[i];
    return S;
}

void sum(const size_t kNumElements) {
    double* A = nullptr;

    warmup<<<1, 1>>>();
    cudaDeviceSynchronize();

    auto t1 = high_resolution_clock::now();

    cudaMallocManaged(&A, kNumElements * sizeof(double));

    auto t2 = high_resolution_clock::now();
    std::cout << "Malloc time: " << duration_cast<milliseconds>(t2 - t1).count() << " ms\n";

    initArray(A, kNumElements);
    auto t3 = high_resolution_clock::now();
    std::cout << "Init time: " << duration_cast<milliseconds>(t3 - t2).count() << " ms\n";

    double* result = nullptr;
    cudaMallocManaged(&result, sizeof(double));

    auto t4 = high_resolution_clock::now();
    std::cout << "Result malloc time: " << duration_cast<milliseconds>(t4 - t3).count() << " ms\n";

    int bpg = ceilDiv(kNumElements);
    atomicSum<<<bpg, THREADS_PER_BLOCK>>>(A, kNumElements, result);
    cudaDeviceSynchronize();

    auto t5 = high_resolution_clock::now();
    std::cout << "Atomic sum time: " << duration_cast<milliseconds>(t5 - t4).count() << " ms\n";

    bool match = (*result == cpuSum(A, kNumElements));
    std::cout << "Atomic Sum Match: " << std::boolalpha << match << "\n";

    auto t6 = high_resolution_clock::now();
    std::cout << "Atomic Sum Match time: " << duration_cast<milliseconds>(t6 - t5).count()
              << " ms\n";

    double* reduce_result = nullptr;
    cudaMallocManaged(&reduce_result, bpg * sizeof(double));

    auto t7 = high_resolution_clock::now();
    std::cout << "Reduce malloc time: " << duration_cast<milliseconds>(t7 - t6).count() << " ms\n";

    reduceSum<<<bpg, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(
            A, kNumElements, reduce_result);
    cudaDeviceSynchronize();
    double reduced_sum = cpuSum(reduce_result, bpg);

    auto t8 = high_resolution_clock::now();
    std::cout << "Reduce Sum time: " << duration_cast<milliseconds>(t8 - t7).count() << " ms\n";

    match = (reduced_sum == cpuSum(A, kNumElements));
    std::cout << "Reduce Sum Match: " << std::boolalpha << match << "\n";

    auto t9 = high_resolution_clock::now();
    std::cout << "Reduce Sum Match time: " << duration_cast<milliseconds>(t9 - t8).count()
              << " ms\n";
}

int main(int argc, char* argv[]) {
    sum(1 << 28);
    return 0;
}