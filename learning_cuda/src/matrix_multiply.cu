#include <cuda_runtime.h>

#include <chrono>
#include <iostream>
#include <thread>
#define THREADS_PER_BLOCK_X 32
#define THREADS_PER_BLOCK_Y 32

using namespace std::chrono;

int ceilDiv(size_t N, size_t tpb) { return (N + tpb - 1) / tpb; }

__global__ void warmup() { return; }

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

__global__ void naiveMatrixMultiplyK(
        const double* A,
        const double* B,
        double* C,
        const size_t kM,
        const size_t kN,
        const size_t kK) {
    int idxX = blockDim.x * blockIdx.x + threadIdx.x;
    int idxY = blockDim.y * blockIdx.y + threadIdx.y;

    if (idxX >= kK || idxY >= kM) return;

    double acc = 0;
    for (int n = 0; n < kN; n++) {
        acc += A[idx(idxY, n, kN)] * B[idx(n, idxX, kK)];
    }
    C[idx(idxY, idxX, kK)] = acc;
}

void naiveMatrixMultiply(
        const double* A,
        const double* B,
        double* C,
        const size_t kM,
        const size_t kN,
        const size_t kK) {
    dim3 bpg(ceilDiv(kK, THREADS_PER_BLOCK_X), ceilDiv(kM, THREADS_PER_BLOCK_Y));
    dim3 tpb(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    naiveMatrixMultiplyK<<<bpg, tpb>>>(A, B, C, kM, kN, kK);
    cudaDeviceSynchronize();
}

// Seems to slower tham naive
__global__ void optimizedMatrixMultiplyK1(
        const double* A,
        const double* B,
        double* C,
        const size_t kM,
        const size_t kN,
        const size_t kK) {
    int tidX = blockDim.x * blockIdx.x + threadIdx.x;
    int tidY = blockDim.y * blockIdx.y + threadIdx.y;

    // 16 KiB of shared memory allocated
    __shared__ double BTile[THREADS_PER_BLOCK_Y][THREADS_PER_BLOCK_X + 1];
    __shared__ double ATile[THREADS_PER_BLOCK_Y][THREADS_PER_BLOCK_X + 1];

    double Ct = 0;

    int numTiles = (kN + THREADS_PER_BLOCK_Y - 1) / THREADS_PER_BLOCK_Y;
    for (int n_ = 0; n_ < numTiles; n_++) {
        // Load contents of B into shared memory
        int By = idx(n_, threadIdx.y, THREADS_PER_BLOCK_Y);
        BTile[threadIdx.y][threadIdx.x] = By < kN ? B[idx(By, tidX, kK)] : 0;

        // Load contents of A into shared memory
        int Ax = idx(n_, threadIdx.x, THREADS_PER_BLOCK_X);
        ATile[threadIdx.y][threadIdx.x] = Ax < kN ? A[idx(tidY, Ax, kN)] : 0;
        __syncthreads();

        // Perform a matrix multiply between ATile and BTile and store in Ct
        for (int j = 0; j < THREADS_PER_BLOCK_X; j++) {
            Ct += ATile[threadIdx.y][j] * BTile[j][threadIdx.x];
        }
        __syncthreads();
    }

    if (tidX >= kK || tidY >= kM) return;

    // Copy from CTile to C
    C[idx(tidY, tidX, kK)] = Ct;
}

void optimizedMatrixMultiply1(
        const double* A,
        const double* B,
        double* C,
        const size_t kM,
        const size_t kN,
        const size_t kK) {
    dim3 bpg(ceilDiv(kK, THREADS_PER_BLOCK_X), ceilDiv(kM, THREADS_PER_BLOCK_Y));
    dim3 tpb(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
    optimizedMatrixMultiplyK1<<<bpg, tpb>>>(A, B, C, kM, kN, kK);
    cudaDeviceSynchronize();
}

bool cpuCheckMultiply(
        const double* A,
        const double* B,
        const double* C,
        const size_t kM,
        const size_t kN,
        const size_t kK) {
    for (size_t i = 0; i < kM; i++) {
        for (size_t j = 0; j < kK; j++) {
            double sum = 0;
            for (size_t k = 0; k < kN; k++) sum += A[idx(i, k, kN)] * B[idx(k, j, kK)];
            if (sum != C[idx(i, j, kK)]) return false;
        }
    }
    return true;
}

void matrixMultiply(const size_t kM, const size_t kN, const size_t kK) {
    warmup<<<1, 1>>>();
    cudaDeviceSynchronize();

    double* A = nullptr;
    double* B = nullptr;
    double* C = nullptr;
    const size_t kA = kM * kN;
    const size_t kB = kN * kK;
    const size_t kC = kM * kK;

    auto t1 = high_resolution_clock::now();

    cudaMalloc(&A, kA * sizeof(double));
    cudaMalloc(&B, kB * sizeof(double));
    cudaMalloc(&C, kC * sizeof(double));

    auto t2 = high_resolution_clock::now();
    std::cout << "Malloc time: " << duration_cast<microseconds>(t2 - t1).count() << " us\n";

    cudaMemset(C, 0, kC * sizeof(double));
    initMatrix(A, kM, kN);
    initMatrix(B, kN, kK);

    auto t3 = high_resolution_clock::now();
    std::cout << "Init time: " << duration_cast<microseconds>(t3 - t2).count() << " us\n";

    naiveMatrixMultiply(A, B, C, kM, kN, kK);

    auto t4 = high_resolution_clock::now();
    std::cout << "Naive Multiply time: " << duration_cast<microseconds>(t4 - t3).count() << " us\n";

    double* A_h = nullptr;
    double* B_h = nullptr;
    double* C_h = nullptr;
    cudaMallocManaged(&A_h, kA * sizeof(double));
    cudaMallocManaged(&B_h, kB * sizeof(double));
    cudaMallocManaged(&C_h, kC * sizeof(double));

    cudaMemcpy(A_h, A, kA * sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(B_h, B, kB * sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(C_h, C, kC * sizeof(double), cudaMemcpyDefault);

    // bool match = cpuCheckMultiply(A_h, B_h, C_h, kM, kN, kK);

    auto t5 = high_resolution_clock::now();

    // std::cout << "Naive Multiply match: " << std::boolalpha << match << "\n";
    std::cout << "Naive Multiply match time: " << duration_cast<microseconds>(t5 - t4).count()
              << " us\n";

    optimizedMatrixMultiply1(A, B, C, kM, kN, kK);

    auto t6 = high_resolution_clock::now();
    std::cout << "Optimized Multiply 1 time: " << duration_cast<microseconds>(t6 - t5).count()
              << " us\n";

    cudaMemcpy(A_h, A, kA * sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(B_h, B, kB * sizeof(double), cudaMemcpyDefault);
    cudaMemcpy(C_h, C, kC * sizeof(double), cudaMemcpyDefault);

    // match = cpuCheckMultiply(A_h, B_h, C_h, kM, kN, kK);

    auto t7 = high_resolution_clock::now();

    // std::cout << "Optimized Multiply 1 match: " << std::boolalpha << match << "\n";
    std::cout << "Optimized Multiply 1 match time: " << duration_cast<microseconds>(t7 - t6).count()
              << " us\n";

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(A_h);
    cudaFree(B_h);
    cudaFree(C_h);
}

int main() {
    matrixMultiply(1 << 13, 1 << 13, 1 << 13);
    return 0;
}
