#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <thread>
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

using namespace std::chrono;

int ceilDiv(size_t N, size_t tpb) { return (N + tpb - 1) / tpb; }

__host__ __device__ size_t idx(const size_t y, const size_t x, const size_t Nx) {
    return y * Nx + x;
}

__global__ void initMatrixK(double* m, const size_t kNy, const size_t kNx) {
    int xid = blockDim.x * blockIdx.x + threadIdx.x;
    int yid = blockDim.y * blockIdx.y + threadIdx.y;

    if (xid >= kNx || yid >= kNy) return;

    m[idx(yid, xid, kNx)] = idx(yid, xid, kNx);
}

void initMatrix(double* m, const size_t kNy, const size_t kNx) {
    dim3 bpg(ceilDiv(kNx, THREADS_PER_BLOCK_X), ceilDiv(kNy, THREADS_PER_BLOCK_Y));
    dim3 tpb(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    initMatrixK<<<bpg, tpb>>>(m, kNy, kNx);
    cudaDeviceSynchronize();
}

__global__ void naiveTransposeK(const double* M, double* M_T, const size_t kNy, const size_t kNx) {
    int xid = blockDim.x * blockIdx.x + threadIdx.x;
    int yid = blockDim.y * blockIdx.y + threadIdx.y;

    if (xid >= kNx || yid >= kNy) return;

    M_T[idx(xid, yid, kNy)] = M[idx(yid, xid, kNx)];
}

void naiveTranspose(const double* M, double* M_T, const size_t kNy, const size_t kNx) {
    dim3 bpg(ceilDiv(kNx, THREADS_PER_BLOCK_X), ceilDiv(kNy, THREADS_PER_BLOCK_Y));
    dim3 tpb(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    naiveTransposeK<<<bpg, tpb>>>(M, M_T, kNy, kNx);
    cudaDeviceSynchronize();
}

__global__ void optimizedTransposeK(
        const double* M, double* M_T, const size_t kNy, const size_t kNx) {
    __shared__ double smem[THREADS_PER_BLOCK_Y][THREADS_PER_BLOCK_X + 1];

    const int tileX = blockDim.x * blockIdx.x;
    const int tileY = blockDim.y * blockIdx.y;

    smem[threadIdx.x][threadIdx.y] = M[idx(tileY + threadIdx.y, tileX + threadIdx.x, kNx)];
    __syncthreads();

    M_T[idx(tileX + threadIdx.y, tileY + threadIdx.x, kNy)] = smem[threadIdx.y][threadIdx.x];
}

void optimizedTranspose(const double* M, double* M_T, const size_t kNy, const size_t kNx) {
    dim3 bpg(ceilDiv(kNx, THREADS_PER_BLOCK_X), ceilDiv(kNy, THREADS_PER_BLOCK_Y));
    dim3 tpb(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    optimizedTransposeK<<<bpg, tpb>>>(M, M_T, kNy, kNx);
    cudaDeviceSynchronize();
}

__global__ void warmup() { return; }

bool checkTranspose(const double* M, const double* M_T, const size_t kNy, const size_t kNx) {
    for (size_t y = 0; y < kNy; y++) {
        for (size_t x = 0; x < kNx; x++) {
            if (abs(M[idx(y, x, kNx)] - M_T[idx(x, y, kNy)]) > 1e-9) return false;
        }
    }
    return true;
}

void matrixTranspose(const size_t kNx, const size_t kNy) {
    warmup<<<1, 1>>>();
    cudaDeviceSynchronize();

    double* M = nullptr;
    double* M_T = nullptr;
    const size_t kN = kNx * kNy;

    auto t1 = high_resolution_clock::now();
    cudaMalloc(&M, kN * sizeof(double));
    cudaMalloc(&M_T, kN * sizeof(double));

    auto t2 = high_resolution_clock::now();
    std::cout << "Malloc time: " << duration_cast<microseconds>(t2 - t1).count() << " us\n";

    cudaMemset(M_T, 0, kN * sizeof(double));
    initMatrix(M, kNy, kNx);

    auto t3 = high_resolution_clock::now();
    std::cout << "Init time: " << duration_cast<microseconds>(t3 - t2).count() << " us\n";

    naiveTranspose(M, M_T, kNy, kNx);

    auto t4 = high_resolution_clock::now();
    std::cout << "Naive Transpose time: " << duration_cast<microseconds>(t4 - t3).count()
              << " us\n";

    double* M_Th = nullptr;
    double* M_h = nullptr;
    cudaMallocManaged(&M_Th, kN * sizeof(double));
    cudaMallocManaged(&M_h, kN * sizeof(double));

    cudaMemcpy(M_Th, M_T, kN * sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(M_h, M, kN * sizeof(double), cudaMemcpyDefault);

    bool match = checkTranspose(M_h, M_Th, kNy, kNx);
    auto t5 = high_resolution_clock::now();

    std::cout << "Naive Transpose Match: " << std::boolalpha << match << "\n";
    std::cout << "Naive Transpose Match time: " << duration_cast<microseconds>(t5 - t4).count()
              << " us\n";

    optimizedTranspose(M, M_T, kNy, kNx);

    auto t6 = high_resolution_clock::now();
    std::cout << "Optimized Transpose time: " << duration_cast<microseconds>(t6 - t5).count()
              << " us\n";

    cudaMemcpy(M_Th, M_T, kN * sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(M_h, M, kN * sizeof(double), cudaMemcpyDefault);

    match = checkTranspose(M_h, M_Th, kNy, kNx);
    auto t7 = high_resolution_clock::now();

    std::cout << "Optimized Transpose Match: " << std::boolalpha << match << "\n";
    std::cout << "Optimized Transpose Match time: " << duration_cast<microseconds>(t7 - t6).count()
              << " us\n";

    cudaFree(M);
    cudaFree(M_T);
    cudaFree(M_Th);
    cudaFree(M_h);
}

int main(int argc, char* argv[]) {
    matrixTranspose(1 << 22, 1 << 6);
    return 0;
}