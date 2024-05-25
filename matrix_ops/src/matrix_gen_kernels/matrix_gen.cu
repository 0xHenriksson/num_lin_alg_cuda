#pragma once

#include "matrix_gen.cuh"



// CUDA error checking function
inline void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(error), file, line);
        exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA_ERROR(error) checkCudaError(error, __FILE__, __LINE__)




// CUDA kernel to generate a random square matrix of given type
template <typename T>
__global__ void gen_rand_square_kernel(T* matrix, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    // generate random values with cuRAND
    if (idx < n && idy < n) {
        curandState state;
        curand_init(1234ULL + idx * n + idy, 0, 0, &state);
        matrix[idx * n + idy] = curand_uniform(&state);
    }
}

template <typename T>
void gen_rand_square_host(T* matrix, int n) {
    T* d_matrix;
    size_t size = n * n * sizeof(T);
    cudaMalloc(&d_matrix, size);
    CHECK_CUDA_ERROR(cudaGetLastError());

    dim3 threadsPerBlock(16, 16);
    int numBlocks = (n + threadsPerBlock.x - 1) / threadsPerBlock.x;
    gen_rand_square_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, n);
    cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaFree(d_matrix);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// wrapper to generate matrix based on type identifier
/*
enum MatrixType {
    INT,
    FLOAT,
    DOUBLE,
    // cutlass numeric types
    HALF_T,
    BFLOAT16_T,
    TFLOAT32_T,
    INT4_T,
    UINT4_T,
    COMPLEX_T
};
*/
void gen_rand_square_matrix(void* matrix, int n, MatrixType type) {

    switch (type) {
        case INT:
            gen_rand_square_host<int>(static_cast<int*>(matrix), n);
            break;
        case FLOAT:
            gen_rand_square_host<float>(static_cast<float*>(matrix), n);
            break;
        case DOUBLE:
            gen_rand_square_host<double>(static_cast<double*>(matrix), n);
            break;
        case HALF_T:
            gen_rand_square_host<__half>(static_cast<__half*>(matrix), n);
            break;
        case BFLOAT16_T:
            gen_rand_square_host<__nv_bfloat16>(static_cast<__nv_bfloat16*>(matrix), n);
            break;
        case TFLOAT32_T:
            gen_rand_square_host<__nv_tfloat32>(static_cast<__nv_tfloat32*>(matrix), n);
            break;
        case INT4_T:
            gen_rand_square_host<__nv_int4>(static_cast<__nv_int4*>(matrix), n);
            break;
        case UINT4_T:
            gen_rand_square_host<__nv_uint4>(static_cast<__nv_uint4*>(matrix), n);
            break;
        case COMPLEX_T:
            gen_rand_square_host<cuComplex>(static_cast<cuComplex*>(matrix), n);
            break;
    }
}

// generate random non-square matrix
// CUDA kernel to generate a random non-square matrix of given type
template <typename T>
__global__ void gen_rand_non_square_kernel(T* matrix, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < m && idy < n) {
        curandState state;
        curand_init(1234ULL + idx * n + idy, 0, 0, &state); // Initialize CURAND on this thread
        matrix[idx * n + idy] = curand_uniform(&state); // Generate random number
    }
}

// Host function to setup kernel launch and handle memory
template <typename T>
void gen_rand_non_square_host(T* matrix, int m, int n) {
    T* d_matrix;
    size_t size = m * n * sizeof(T);
    cudaMalloc(&d_matrix, size); // Allocate memory on device
    CHECK_CUDA_ERROR(cudaGetLastError());

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    gen_rand_non_square_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, m, n); // Launch kernel
    cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost); // Copy result back to host
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaFree(d_matrix); // Free device memory
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// Wrapper function to handle different data types
void gen_rand_non_square(void* matrix, int m, int n, MatrixType type) {
    switch (type) {
        case INT:
            gen_rand_non_uniform<int>(static_cast<int*>(matrix), m, n);
            break;
        case FLOAT:
            gen_rand_non_uniform<float>(static_cast<float*>(matrix), m, n);
            break;
        case DOUBLE:
            gen_rand_non_uniform<double>(static_cast<double*>(matrix), m, n);
            break;
        case HALF_T:
            gen_rand_non_uniform<__half>(static_cast<__half*>(matrix), m, n);
            break;
        case BFLOAT16_T:
            gen_rand_non_uniform<__nv_bfloat16>(static_cast<__nv_bfloat16*>(matrix), m, n);
            break;
        case TFLOAT32_T:
            gen_rand_non_uniform<__nv_tfloat32>(static_cast<__nv_tfloat32*>(matrix), m, n);
            break;
        case INT4_T:
            gen_rand_non_uniform<__nv_int4>(static_cast<__nv_int4*>(matrix), m, n);
            break;
        case UINT4_T:
            gen_rand_non_uniform<__nv_uint4>(static_cast<__nv_uint4*>(matrix), m, n);
            break;
        case COMPLEX_T:
            gen_rand_non_uniform<cuComplex>(static_cast<cuComplex*>(matrix), m, n);
            break;
    }
}



// generate null/zero matrix (square or non-square) kernel
template <typename T>
__global__ void gen_null_kernel(T* matrix, int m, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // set all values of the matrix to 0
    if (idx < m && idy < n) {
        matrix[idx * n + idy] = 0;
    }
}

//host function generate null/zero matrix (square or non-square) 
template <typename T>
void gen_null_host(T* matrix, int m, int n) {
    T* d_matrix;
    size_t size = m * n * sizeof(T);
    cudaMalloc(&d_matrix, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    gen_null_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, m, n);
    cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaFree(d_matrix);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// wrapper function to handle different data types
void gen_null_matrix(void* matrix, int m, int n, MatrixType type) {
    switch (type) {
        case INT:
            gen_null_host(static_cast<int*>(matrix), m, n);
            break;
        case FLOAT:
            gen_null_host(static_cast<float*>(matrix), m, n);
            break;
        case DOUBLE:
            gen_null_host(static_cast<double*>(matrix), m, n);
            break;
        case HALF_T:
            gen_null_host(static_cast<__half*>(matrix), m, n);
            break;
        case BFLOAT16_T:
            gen_null_host(static_cast<__nv_bfloat16*>(matrix), m, n);
            break;
        case TFLOAT32_T:
            gen_null_host(static_cast<__nv_tfloat32*>(matrix), m, n);
            break;
        case INT4_T:
            gen_null_host(static_cast<__nv_int4*>(matrix), m, n);
            break;
        case UINT4_T:
            gen_null_host(static_cast<__nv_uint4*>(matrix), m, n);
            break;
        case COMPLEX_T:
            gen_null_host(static_cast<cuComplex*>(matrix), m, n);
            break;
    }
}

// generate identity matrix (square) kernel
template <typename T>
__global__ void gen_identity_kernel(T* matrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    // assign 1 to the diagonal elements, 0 to the rest
    if (idx < n && idy < n) {
        matrix[idx * n + idy] = (idx == idy) ? 1 : 0;
    }
}

// host generate identity matrix
template <typename T>
void gen_identity_host(T* matrix, int n) {
    T* d_matrix;
    size_t size = n * n * sizeof(T);
    cudaMalloc(&d_matrix, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    gen_identity_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, n);
    cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaFree(d_matrix);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// wrapper function to handle different data types
void gen_identity_matrix(void* matrix, int n, MatrixType type) {
    switch (type) {
        case INT:
            gen_identity_host(static_cast<int*>(matrix), n);
            break;
        case FLOAT:
            gen_identity_host(static_cast<float*>(matrix), n);
            break;
        case DOUBLE:
            gen_identity_host(static_cast<double*>(matrix), n);
            break;
        case HALF_T:
            gen_identity_host(static_cast<__half*>(matrix), n);
            break;
        case BFLOAT16_T:
            gen_identity_host(static_cast<__nv_bfloat16*>(matrix), n);
            break;
        case TFLOAT32_T:
            gen_identity_host(static_cast<__nv_tfloat32*>(matrix), n);
            break;
        case INT4_T:
            gen_identity_host(static_cast<__nv_int4*>(matrix), n);
            break;
        case UINT4_T:
            gen_identity_host(static_cast<__nv_uint4*>(matrix), n);
            break;
        case COMPLEX_T:
            gen_identity_host(static_cast<cuComplex*>(matrix), n);
            break;
    }
}


/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
Matrix conversion kernels
- take in randomly generated matrix
- perform conversion in device
* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// convert to lower triangular on device
__global__ void convert_to_lower_triangular_kernel(T* matrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < n && idy < n) {
        matrix[idx * n + idy] = (idx >= idy) ? matrix[idx * n + idy] : 0;
    }
}

// host function to call convert to lower triangular kernel
void convert_to_lower_triangular(void* matrix, int n) {
    int* d_matrix;
    size_t size = n * n * sizeof(T);
    cudaMalloc(&d_matrix, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    convert_to_lower_triangular_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, n);
    cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaFree(d_matrix);
    CHECK_CUDA_ERROR(cudaGetLastError());
}


// convert to upper triangular on device
__global__ void convert_to_upper_triangular_kernel(T* matrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < n && idy < n) {
        matrix[idx * n + idy] = (idx <= idy) ? matrix[idx * n + idy] : 0;
    }
}

// host function to call convert to upper triangular kernel
void convert_to_upper_triangular(void* matrix, int n) {
    T* d_matrix;
    size_t size = n * n * sizeof(T);
    cudaMalloc(&d_matrix, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    convert_to_upper_triangular_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, n);
    cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaFree(d_matrix);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// convert to diagonal matrix kernel
__global__ void convert_to_diagonal_kernel(T* matrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < n && idy < n) {
        matrix[idx * n + idy] = (idx == idy) ? matrix[idx * n + idy] : 0;
    }
}

// host function to call convert to diagonal kernel
void convert_to_diagonal(void* matrix, int n) {
    T* d_matrix;
    size_t size = n * n * sizeof(T);
    cudaMalloc(&d_matrix, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    convert_to_diagonal_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, n);
    cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaFree(d_matrix);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// convert to symmetric matrix kernel by computing the transpose
__global__ void convert_to_symmetric_kernel(T* matrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < n && idy < n) {
        matrix[idx * n + idy] = (idx <= idy) ? matrix[idx * n + idy] : matrix[idy * n + idx];
    }
}

// host function to call convert to symmetric kernel
void convert_to_symmetric(void* matrix, int n) {
    T* d_matrix;
    size_t size = n * n * sizeof(T);
    cudaMalloc(&d_matrix, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    convert_to_symmetric_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, n);
    cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaFree(d_matrix);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// convert to skew symmetric matrix kernel
__global__ void convert_to_skew_symmetric_kernel(T* matrix, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < n && idy < n) {
        matrix[idx * n + idy] = (idx <= idy) ? matrix[idx * n + idy] : -matrix[idy * n + idx];
    }
}

// host function to call convert to skew symmetric kernel
void convert_to_skew_symmetric(void* matrix, int n) {
    T* d_matrix;
    size_t size = n * n * sizeof(T);
    cudaMalloc(&d_matrix, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    convert_to_skew_symmetric_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, n);
    cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaFree(d_matrix);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

template <typename T>
__global__ void initCurandStates(curandState* states, int n) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        curand_init(1234ULL + id, 0, 0, &states[id]);
    }
}

template <typename T>
__global__ void generateRandomMatrix(T* matrix, int n, curandState* states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < n && idy < n) {
        matrix[idx * n + idy] = curand_normal(&states[threadIdx.x]);
    }
}


// CUDA kernel to perform the Modified Gram-Schmidt process for QR decomposition
template <typename T>
__global__ void modifiedGramSchmidt(T* matrix, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n) return;

    // Temporary variables for vector norms and dot products
    T r_kk, dot_product;
    // Pointer to the k-th column
    T* v_k = matrix + k * n;
    // Compute the norm of the k-th column vector
    r_kk = 0;
    for (int i = 0; i < n; i++) {
        r_kk += v_k[i] * v_k[i];
    }
    r_kk = sqrt(r_kk);
    // Normalize the k-th column vector
    for (int i = 0; i < n; i++) {
        v_k[i] /= r_kk;
    }
    // Orthogonalize all subsequent columns
    __syncthreads(); // Synchronize threads to ensure k-th column is updated
    for (int j = k + 1; j < n; j++) {
        // Pointer to the j-th column
        T* v_j = matrix + j * n;
        // Compute the dot product of v_k and v_j
        dot_product = 0;
        for (int i = 0; i < n; i++) {
            dot_product += v_k[i] * v_j[i];
        }
        // Subtract the projection of v_j onto v_k
        for (int i = 0; i < n; i++) {
            v_j[i] -= dot_product * v_k[i];
        }
    }
}

// Host function to orchestrate the generation of an orthonormal matrix
template <typename T>
void generateOrthonormalMatrix(T* matrix, int n) {
    T* d_matrix;
    curandState* d_states;
    cudaMalloc(&d_matrix, n * n * sizeof(T));
    cudaMalloc(&d_states, n * sizeof(curandState));

    dim3 threadsPerBlock(256);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x);

    initCurandStates<<<numBlocks, threadsPerBlock>>>(d_states, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    generateRandomMatrix<<<numBlocks, threadsPerBlock>>>(d_matrix, n, d_states);
    CHECK_CUDA_ERROR(cudaGetLastError());
    modifiedGramSchmidt<<<numBlocks, threadsPerBlock>>>(d_matrix, n);
    CHECK_CUDA_ERROR(cudaGetLastError());

    cudaMemcpy(matrix, d_matrix, n * n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaFree(d_matrix);
    cudaFree(d_states);
}

// convert to orthogonal matrix kernel
// requires input matrix is orthonormal
// orthogonal matrix is a matrix whose inverse is equal to its transpose
// A^{-1} = A^T
__global__ void convert_to_orthogonal_kernel(T* matrix, int n) {
    T* d_matrix;
    size_t size = n * n * sizeof(T);
    cudaMalloc(&d_matrix, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    // compute the orthogonal matrix
    // Orthogonal matrix computation requires the matrix to be square and the rows to be orthonormal.
    // This kernel will assume the input matrix is already on the device and square.
    // The output will be the transpose of the matrix since for orthogonal matrices A^T = A^-1.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n && idy < n) {
        // Transpose the matrix: swap rows and columns
        d_matrix[idy * n + idx] = matrix[idx * n + idy];
    }
    // Note: This simplistic approach assumes the input matrix is already orthonormal.
    // In a full implementation, additional steps would be required to orthonormalize the matrix.
}

// host function to convert to orthogonal matrix
void convert_to_orthogonal(void* matrix, int n) {
    T* d_matrix;
    size_t size = n * n * sizeof(T);
    cudaMalloc(&d_matrix, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // compute the orthogonal matrix
    convert_to_orthogonal_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, n);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaMemcpy(matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaFree(d_matrix);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

// convert random matrix to unitary matrix kernel
__global__ void convertToUnitaryKernel()

// convert to hessenberg

// 




