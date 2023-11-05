#include "cublas-MatMul2D.cuh"
#include "cublas_utils.cuh"

int MatMul2D(OUT float* C, IN const float* A, const float* B, int M, int N, int K)
{
    // Initialize CUDA and cuBLAS
    cudaError_t cudaStat;
    cublasStatus_t stat;
    cublasHandle_t handle;

    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS initialization failed" << std::endl;
        return 1;
    }

    // Allocate and copy matrices A and B to the GPU
    float* d_A, * d_B, * d_C;
    cudaStat = cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaStat = cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaStat = cudaMalloc((void**)&d_C, M * N * sizeof(float));
    cudaStat = cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaStat = cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Perform GEMM operation
    float alpha = 1.0f;
    float beta = 0.0f;

    // use CUBLAS_OP_T due to rowmajor tensor and columnmajor cuda

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, d_A, K, d_B, N, &beta, d_C, M));

    // Copy the result back to the CPU
    cudaStat = cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
#if 0
    // Display the result
    std::cout << "Result Matrix C:" << std::endl;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
#endif
    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}