#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

// Parâmetros da matriz
const int N = 20000;          // Matriz N x N
const int NNZ = 2000000;      // 2 milhões de elementos não-nulos (~0.5%)

int main() {
    // Gerador aleatório
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> col_dist(0, N - 1);
    std::uniform_real_distribution<float> val_dist(-1.0f, 1.0f);

    // Alocar estruturas CSR na CPU
    std::vector<int> h_csrRowPtr(N + 1, 0);
    std::vector<int> h_csrColInd;
    std::vector<float> h_csrVal;

    int nnz_per_row = NNZ / N;

    for (int i = 0; i < N; ++i) {
        h_csrRowPtr[i + 1] = h_csrRowPtr[i] + nnz_per_row;
        for (int j = 0; j < nnz_per_row; ++j) {
            h_csrColInd.push_back(col_dist(rng));
            h_csrVal.push_back(val_dist(rng));
        }
    }

    // Vetor x e y
    std::vector<float> h_x(N, 1.0f);
    std::vector<float> h_y(N, 0.0f);

    // Alocar GPU
    int *d_csrRowPtr, *d_csrColInd;
    float *d_csrVal, *d_x, *d_y, *dBuffer;
    cudaMalloc(&d_csrRowPtr, (N + 1) * sizeof(int));
    cudaMalloc(&d_csrColInd, NNZ * sizeof(int));
    cudaMalloc(&d_csrVal, NNZ * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));

    // Copiar dados para GPU
    cudaMemcpy(d_csrRowPtr, h_csrRowPtr.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, h_csrColInd.data(), NNZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal, h_csrVal.data(), NNZ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Criar handle cuSPARSE
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Criar descritores
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateCsr(&matA, N, N, NNZ, d_csrRowPtr, d_csrColInd, d_csrVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecX, N, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, N, d_y, CUDA_R_32F);

    // Buffer temporário
    size_t bufferSize;
    float alpha = 1.0f, beta = 0.0f;
    cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSize);
    cudaMalloc(&dBuffer, bufferSize);

    // Cronometrar execução massiva
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100000; ++i) {
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
            CUSPARSE_MV_ALG_DEFAULT, dBuffer);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Tempo para 100000 SpMV grandes: " << elapsed.count() << " segundos\n";

    // Limpeza
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    cudaFree(d_csrRowPtr); cudaFree(d_csrColInd); cudaFree(d_csrVal);
    cudaFree(d_x); cudaFree(d_y); cudaFree(dBuffer);

    return 0;
}