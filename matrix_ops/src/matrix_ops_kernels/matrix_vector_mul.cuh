// Kernel 1: Matrix-Vector Multiplication

// only include once in a single compilation
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>


void inline cublasCheck(cublasStatus_t result) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d\n", result);
        assert(result == CUBLAS_STATUS_SUCCESS);
    }
}

/*
@matrix: matrix to multiply
@vector: vector to multiply
@result: result of the multiplication
@rows: number of rows in the matrix
@cols: number of columns in the matrix
*/
template <typename T>
void matrixVectorMul(const T* matrix, const T* vector, T* result, int rows, int cols) {

    // create cuBlAS handle
    cublasHandle_t handle;
    cublasCheck(cublasCreate);

    // set pointer mode to device
    cublasCheck(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));

    // perform matrix-vector multiplication
    const T alpha = 1.0
    const T beta = 0.0;
    cublasCheck(cublasSgemv(handle, CUBLAS_OP_N, rows, cols, &alpha, matrix, rows, vector, 1, &beta, result, 1));

    cublasCheck(cublasDestroy(handle));
}

// kernel to launch matrix-vector multiplication
teplate <typename T>
__global__ void matrixVectorMulKernel(const T* matrix, const T* vector, T* result,
                                      int rows, int cols) {

    if (cols != rows) {
        std::cerr << "Error: matrix and vector dimensions do not match\n" << std::endl;
        return;
    }

    // call the matrix-vector multiplication function
    matrixVectorMul(matrix, vector, result, rows, cols);
}

