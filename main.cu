#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

using namespace std;

__global__ void cuda_hello() {
    // Instructions have to be in C in GPU instructions
    printf("Hello World from GPU!\n");
}

int main() {
    // blocks in a grid, number of threads in a block\
    // grid, block
    cuda_hello <<<1, 1>>> ();
    cudaDeviceSynchronize(); // Wait for the GPU to finish
    cout << "Hello World from CPU!" << endl;
    return 0;
}
