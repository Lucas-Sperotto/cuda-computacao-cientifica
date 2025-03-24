#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <chrono>

#define CHECK_CUDA(call) \
    if ((call) != cudaSuccess) { \
        std::cerr << "CUDA error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CHECK_CUBLAS(call) \
    if ((call) != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main() {
    const size_t gpu_mem_bytes = 4096UL * 1024 * 1024; // 4096 MiB
    const size_t float_size = sizeof(float);

    // Três matrizes: A (MxK), B (KxN), C (MxN) → A * B = C
    // Para usar toda a memória, resolvemos: 3 * M * K * sizeof(float) ≈ gpu_mem_bytes
    // Assumindo M = N = K
    size_t max_elements = gpu_mem_bytes / (3 * float_size);
    size_t N = static_cast<size_t>(sqrt(max_elements));

    size_t num_bytes_matrix = N * N * float_size;
    std::cout << "Usando matrizes de tamanho " << N << " x " << N << " (~" << (3.0 * num_bytes_matrix / (1024 * 1024)) << " MiB)\n";

    // Timers
    std::chrono::high_resolution_clock::time_point start, end;
    std::chrono::duration<double> t_create, t_alloc, t_copy, t_compute;

    // Etapa 1: Criação no host
    start = std::chrono::high_resolution_clock::now();
    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *h_C = new float[N * N];
    for (size_t i = 0; i < N * N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    end = std::chrono::high_resolution_clock::now();
    t_create = end - start;

    // Etapa 2: Alocação na GPU
    start = std::chrono::high_resolution_clock::now();
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, num_bytes_matrix));
    CHECK_CUDA(cudaMalloc(&d_B, num_bytes_matrix));
    CHECK_CUDA(cudaMalloc(&d_C, num_bytes_matrix));
    end = std::chrono::high_resolution_clock::now();
    t_alloc = end - start;

    // Etapa 3: Transferência para GPU
    start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA(cudaMemcpy(d_A, h_A, num_bytes_matrix, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, num_bytes_matrix, cudaMemcpyHostToDevice));
    end = std::chrono::high_resolution_clock::now();
    t_copy = end - start;

    // Etapa 4: Computação (A x B = C)
    start = std::chrono::high_resolution_clock::now();
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             d_B, N,
                             d_A, N,
                             &beta,
                             d_C, N));
    CHECK_CUDA(cudaDeviceSynchronize());
    end = std::chrono::high_resolution_clock::now();
    t_compute = end - start;

    // Resultados
    std::cout << "\n==== TEMPOS (em segundos) ====" << std::endl;
    std::cout << "Criação das matrizes (host):\t" << t_create.count() << " s\n";
    std::cout << "Alocação na GPU:\t\t" << t_alloc.count() << " s\n";
    std::cout << "Transferência para GPU:\t" << t_copy.count() << " s\n";
    std::cout << "Multiplicação (GPU):\t\t" << t_compute.count() << " s\n";

    // Liberação
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}
