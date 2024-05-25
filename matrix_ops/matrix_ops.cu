#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <driver.cuh>
#include <vector>
#include "matrix_ops.cuh"

#define cudaCheck(err) __cudaCheck(err, __FILE__, __LINE__)

/*
args:
    1. device number
    2. kernel number
    3. .txt or .csv file with matrix/vector data
*/
int main(int argc, char **argc) {

    const int available_kernels = 1;
    

    // get kernel to test with error check
    if (argc != 2) { // if no kernel is specified
        // list kernels to demonstrate
        // void print_kernels();
        printf("Kernels:\n
            1. Matrix times a vector\n");
        std::cerr << "Please select a kernel"
                << std::endl;
        exit(EXIT_FAILURE);
    }
    int kernel_num = std::stoi(argv[1]);
    if (kernel_num < 0 || kernel_num > available_kernels) {
        printf("Please enter a valid kernel number (0-%d)", available_kernels);
        exit(EXIT_FAILURE);
    }

    // if no input file is specified, generate random data depending on kernel selection

    // get environment variable for device
    int deviceIdx = 0;
    if (getenv("DEVICE") != NULL) {
        deviceIdx = atoi(getenv("DEVICE"));
    }
    cudaCheck(cudaSetDevice(deviceIdx));

    printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

    // generate matrices/vectors/inputs with respect to kernel
    switch (kernel_num) {
        // matrix times a vector
        case 1:
            // compute the result
            run_matrix_vector_mul(matrix, vector);
            break;
        // matrix times a matrix
        case 2:
            // compute the result
            run_matrix_matrix_mul(matrix, matrix);
            break;
    }

}