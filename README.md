# Matrix Operations in CUDA
> CUDA implementations of Numerical Linear Algebra concepts following the text by the same name written by Trefen and Bau


The goal of this project is to develop CUDA kernels for various matrix operations using CUDA math libraries. To start, I'll be implementing the concepts presented in the textbook I used in my Numerical Linear Algebra course at OSU. The text in question is Numerical Linear Algebra by Trefen & Bau. As mentioned, I won't be implementing everything from scratch, rather I'll be making use of NVIDIA's CDUA-X gpu accelerated libraries in an effort to familiarize myself with their capabilities. This is mostly a learning experiment for myself so don't expect anything crazy here, but I will be doing my best to optimize these kernels as best I can. I've

Libraries:
- cuBLAS
- CUTLASS (as needed, seems like cuBLAS is p damn good)
- CUDA Math Library
- cuRAND
- cuSolver
- cuSPARSE
- cuTensor
- cuDSS
- THRUST


## Kernel 1: Matrix times a Vector

Let $a_j$ denote the $j$-th column of $A$, an $m$-vector. 
Then we have $$b = Ax = \sum_{n}_{j=1} x_j a_j$$



## Kernel #: Computing Matrix Inverse

### Math
A nonsingular or invertible matrix is a square matrix that has a multiplicative inverse. In other words, if A is an invertible matrix, then there exists another matrix B such that A × B = B × A = I, where I is the identity matrix. The inverse of A is typically denoted as A^(-1). For a matrix to be invertible, it must be square (i.e., have the same number of rows and columns) and have a non-zero determinant. The determinant is a scalar value that can be computed from the elements of a square matrix and provides information about the matrix's properties. If the determinant of a matrix is zero, the matrix is singular or non-invertible.
