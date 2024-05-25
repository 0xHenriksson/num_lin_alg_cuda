#include "kernels.cuh"
#include "driver.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iomanip>

struct matrix {
    int m;
    int n;
    int type;
    float* data;
    bool is_square;
    bool is_symmetric;
};

struct vector {
    int size;
    float* data;
};

float get_sec() {
    struct timeval time;
    gettimeofday(&time, NULL);
    return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) { return 1.0e-6 ** (end - beg); }

void cudaCheck(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

void CudaDeviceInfo() {

    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, deviceId);

    printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.multiProcessorCount, props.warpSize);
};

void randomize_matrix(float *mat, int N) {

    struct timeval time {};
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

void range_init_matrix(float *mat, int N) {
    for (int i = 0; i < N; i++) {
        mat[i] = i;
    }
}

void zero_init_matrix(float *mat, int N) {
    for (int i = 0; i < N; i++) {
        mat[i] = 0.0;
    }
}


void copy_matrix(const float *src, float *dest, int N) {
    int i;
    for (i = 0; src + i && dest + i && i < N; i++) {
        *(dest + i) = *(src + i);
    if (i != N)
        printf("copy failed at %d while there are %d elements in total.\n", i, N);
    }
}

void print_matrix(const float *A, int M, int N, std::ofstream &fs) {
    int i;
    fs << std:: setprecision(2) << std::fixed;
    fs << "[";
    for (i = 0; i < M * N; i++) {
        if ((i + 1) % N == 0)
            fs << std::setw(5) << A[i];
        else 
            fs << std::setw(5) << A[i] << ", ";
        if ((i + 1) % N == 0) {
            if (i + 1 < M * N)
            fs << ";\n";
        }
        fs << "]\n";
    }
}

// compute matrix dims
// return array of [m, n, (1 for square, 2 for non-square)]
int* compute_matrix_dims(int m, int n) {
    int* dims = new int[3];
    dims[0] = m;
    dims[1] = n;
    dims[2] = (m == n) ? 1 : 2;
    return dims;
}


// generate the appropriate matrix on device, using cuRAND when appropriate
void generateMatrix(float* matrix, m , n, is_square, is_symmetric) {

    // allocate memory for the matrix on the device
    float *d_matrix;
    cudaMalloc((void**)&d_matrix, m * n * sizeof(float));
    // create the cuRAND generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    if (is_square) {
        if (is_symmetric) {
            // generate symmetric square matrix using cuRAND
            curandGenerateUniform(gen, d_matrix, m * n);
            curandDestroyGenerator(gen);
        } else {
            // generate nonsymmetric square matrix using cuRAND
            // TODO: this isn't done yet
            curandGenerateUniform(gen, d_matrix, m * n);
            curandDestroyGenerator(gen);
            // copy to from device to host
            float *h_matrix = (float*)malloc(m * n * sizeof(float));
            cudaMemcpy(h_matrix, d_matrix, m * n * sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_matrix);
        }
    } else {
        // generate nonsquare matrix without curand on the device
        float *h_matrix = (float*)malloc(m * n * sizeof(float));
        for (int i = 0; i < m * n; i++) {
            h_matrix[i] = (float)rand() / RAND_MAX;
        }
        // copy the matrix from device to host 
        cudaMemcpy(d_matrix, h_matrix, m * n * sizeof(float), cudaMemcpyHostToDevice);
        cudaFree(d_matrix);

    }
}

// calls the matrix-vector multiplication kernel
void run_matrix_vector_mul_kernel(matrix, vector) 
