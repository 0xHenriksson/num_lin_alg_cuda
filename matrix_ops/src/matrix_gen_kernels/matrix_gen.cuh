#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

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

/* * * * * * * * * * * * * * * * * * * * * * 
these kernels generate the starting point to later be converted
kernels use cutlass numeric types
kernels can generate different cutlass types depending on args
kernels generate random values using cuRAND
* * * * * * * * * * * * * * * * * * * * * * * * */
// generate random square matrix (NxN)
template <typename T> __global__ void gen_rand_square_matrix(float* matrix, int n)
// wrapper to generate matrix based on type identifier
void gen_rand_square_matrix_with_type(void* matrix, int n, MatrixType type)
// generate random non-square matrix (MxN)
void gen_rand_non_square(void* matrix, int m, int n, MatrixType type)
// generate null/zero matrix (square or non-square)
void gen_null_kernel(void* matrix, int m, int n, MatrixType type)
void gen_null_matrix(void* matrix, int m, int n, MatrixType type)
// generate identity matrix (NxN)
void gen_identity_matrix(void* matrix, int n, MatrixType type)

/* * * * * * * * * * * * * * * * * * * * * * 
these kernels operate on the random matrix to convert it to the desired type
kernels use cutlass numeric types

* * * * * * * * * * * * * * * * * * * * * * * * */
// 
// generate triangular matrix
// generate diagonal matrix
// transpose matrix
// sy