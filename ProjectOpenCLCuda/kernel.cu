#ifndef KERNEL_CUH
#define KERNEL_CUH


#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>

#include "common.h"


/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
template <int BLOCK_SIZE> __global__ void MatrixMulCUDA(float *C, float *A,
                                                        float *B, int wA,
                                                        int wB) {
    // Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    // Index of the last sub-matrix of A processed by the block
    int aEnd   = aBegin + wA - 1;

    // Step size used to iterate through the sub-matrices of A
    int aStep  = BLOCK_SIZE;

    // Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    // Step size used to iterate through the sub-matrices of B
    int bStep  = BLOCK_SIZE * wB;

    // Csub is used to store the element of the block sub-matrix
    // that is computed by the thread
    double Csub = 0;

    // Loop over all the sub-matrices of A and B
    // required to compute the block sub-matrix
    for (int a = aBegin, b = bBegin;
            a <= aEnd;
            a += aStep, b += bStep) {
        // Declaration of the shared memory array As used to
        // store the sub-matrix of A
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];

        // Declaration of the shared memory array Bs used to
        // store the sub-matrix of B
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load the matrices from device memory
        // to shared memory; each thread loads
        // one element of each matrix
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];

        // Synchronize to make sure the matrices are loaded
        __syncthreads();

        // Multiply the two matrices together;
        // each thread computes one element
        // of the block sub-matrix
#pragma unroll

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Csub += As[ty][k] * Bs[k][tx];
        }

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write the block sub-matrix to device memory;
    // each thread writes one element
    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = (float) Csub;
}


/**
 * Run a simple test of matrix multiplication using CUDA
 */
 extern "C" void MatrixMultiply(float* h_A, float* h_B, float *gpu_C, int block_size, Configuration configuration) {
   
    // Allocate device memory
    float *d_A, *d_B, *d_C;


    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_A), configuration.mem_size_A));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_B), configuration.mem_size_B));

    checkCudaErrors(cudaMalloc(reinterpret_cast<void **>(&d_C), configuration.mem_size_C));

    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, configuration.mem_size_A, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_B, h_B, configuration.mem_size_B, cudaMemcpyHostToDevice));

    // Setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(configuration.WC / threads.x, configuration.HC / threads.y);

    // Create and start timer
    printf("Computing result using CUDA Kernel...\n");

    startTimer();

    // Execute the kernel
    if (block_size == 16) {
        MatrixMulCUDA<16> <<< grid, threads >>>(d_C, d_A, d_B, configuration.WC, configuration.HC);
    } else {
         MatrixMulCUDA<32> <<< grid, threads >>>(d_C, d_A, d_B, configuration.WC, configuration.HC);
    }
    
    stopTimer();

    // Copy result from device to host
    checkCudaErrors(cudaMemcpy(gpu_C, d_C, configuration.mem_size_C, cudaMemcpyDeviceToHost));


    // Clean up memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));


    
}








#endif
