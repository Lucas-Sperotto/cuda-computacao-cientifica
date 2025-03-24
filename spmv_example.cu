#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>

int main() {
    // Exemplo: matriz 3x3 esparsa em CSR
    int rows = 3, cols = 3, nnz = 4;

    // CSR representation (host)
    int h_csrRowPtr[] = {0, 1, 3, 4};      // tamanho = n_rows + 1
    int h_csrColInd[] = {0, 0, 2, 1};      // tamanho = nnz
    float h_csrVal[]  = {10, 20, 30, 40};  // tamanho = nnz

    float h_x[] = {1, 2, 3};               // vetor de entrada
    float h_y[3];                          // resultado

    // Alocar memória na GPU
    int *d_csrRowPtr, *d_csrColInd;
    float *d_csrVal, *d_x, *d_y;

    cudaMalloc((void**)&d_csrRowPtr, (rows + 1) * sizeof(int));
    cudaMalloc((void**)&d_csrColInd, nnz * sizeof(int));
    cudaMalloc((void**)&d_csrVal, nnz * sizeof(float));
    cudaMalloc((void**)&d_x, cols * sizeof(float));
    cudaMalloc((void**)&d_y, rows * sizeof(float));

    // Copiar dados para GPU
    cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColInd, h_csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrVal, h_csrVal, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, cols * sizeof(float), cudaMemcpyHostToDevice);

    // Criar handle cuSPARSE
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Criar descritor da matriz esparsa
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateCsr(&matA, rows, cols, nnz,
                      d_csrRowPtr, d_csrColInd, d_csrVal,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    // Vetores densos
    cusparseCreateDnVec(&vecX, cols, d_x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, rows, d_y, CUDA_R_32F);

    // Buffer temporário
    size_t bufferSize;
    void* dBuffer = nullptr;
    float alpha = 1.0f, beta = 0.0f;

    cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, &bufferSize);

    cudaMalloc(&dBuffer, bufferSize);

    // Executar SpMV
    cusparseSpMV(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, vecX, &beta, vecY, CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT, dBuffer);

    // Copiar resultado para CPU
    cudaMemcpy(h_y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Resultado y = A*x:\n";
    for (int i = 0; i < rows; ++i) std::cout << h_y[i] << " ";
    std::cout << "\n";

    // Liberar memória e handles
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
    cudaFree(d_csrRowPtr); cudaFree(d_csrColInd); cudaFree(d_csrVal);
    cudaFree(d_x); cudaFree(d_y); cudaFree(dBuffer);

    return 0;
}
