/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix Addition: C = A + B.
 * Host code.
 */

// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>


// #define N (256*84)
#define N (32*62)


// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N], int type) {
    
    int i,j;
    if (type == 1)
    {
        int linearIndex = blockIdx.x*blockDim.x+threadIdx.x;
        i= linearIndex / N;
        j= linearIndex % N;

        // if (linearIndex < 30)
        //     printf ("(%d,%d) = %d \n", i,j, linearIndex);
    }
    else if (type == 2)
    {
        i = blockIdx.x * blockDim.x + threadIdx.x;
        j = blockIdx.y * blockDim.y + threadIdx.y;
    }
    else if (type == 3)
    {

        int blockId  = blockIdx.x + blockIdx.y * gridDim.x;
        // j = blockId * blockDim.x + threadIdx.x;
        // i = blockId * blockDim.y + threadIdx.y;
        int linearIndex = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
        i= linearIndex / N;
        j= linearIndex % N;

        // if (linearIndex < 30)
        //     printf ("(%d,%d) = %d \n", i,j, linearIndex);
    }

    
    if (i < N && j < N)
    {
        C[i][j] = A[i][j] + B[i][j];
    }
}


void printMatrix (float* A)
{
    printf("\n");
    for (int i = 0; i < N*N; ++i)
    {       
        printf("%f ", A[i]);
        
            
    }
}
/**
 * Program main
 */


int main(int argc, char **argv) {

    cudaError_t err = cudaSuccess;
    float *h_A, *h_B, *h_C;
    
    
    float (*d_A)[N]; //pointers to arrays of dimension N
    float (*d_B)[N];
    float (*d_C)[N];

    h_A = new float[N*N];
    h_B = new float[N*N];
    h_C = new float[N*N];

    int numElements = N*N;
    size_t size = numElements * sizeof(float);

    for(int i = 0; i < N*N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;;
        h_B[i] = rand() / (float)RAND_MAX;;
        
    }  

    int blockSize = strtol(argv[1], NULL, 10);
    int type =strtol(argv[2], NULL, 10);

    printf("[Matrix addition of %d elements] size is %d\n", numElements, size);

    //allocation
    checkCudaErrors(cudaMalloc((void**)&d_A, (N*N)*sizeof(h_A[0])));
    checkCudaErrors(cudaMalloc((void**)&d_B, (N*N)*sizeof(h_B[0])));
    checkCudaErrors(cudaMalloc((void**)&d_C, (N*N)*sizeof(h_C[0])));


    //copying from host to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, (N*N)*sizeof(h_A[0]), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, (N*N)*sizeof(h_B[0]), cudaMemcpyHostToDevice));



    dim3 threadsPerBlock;
    dim3 numBlocks;

    printf("type: %d \n ", type);


    // Launch the Vector Add CUDA Kernel

    if (type == 1) //1D - 1D
    {
        threadsPerBlock.x = blockSize;
        numBlocks.x = (numElements + threadsPerBlock.x - 1) / threadsPerBlock.x;
    }
    else if (type == 2) //2D - 2D
    {
        threadsPerBlock = dim3(blockSize, blockSize);
        int gridX = (sqrt(numElements) + threadsPerBlock.x - 1) / threadsPerBlock.x;
        int gridY = (sqrt(numElements) + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks = dim3(gridX, gridY);
    }

    else if (type == 3) //2D - 1D
    {
        // threadsPerBlock = dim3(blockSize, 1);
        // int gridX = (sqrt(numElements) + threadsPerBlock.x - 1) / threadsPerBlock.x;
        // int gridY = (sqrt(numElements) + threadsPerBlock.y - 1) / threadsPerBlock.y;
        // numBlocks = dim3(gridX, gridY);

        threadsPerBlock.x = blockSize;
        int blocksPerGrid =(numElements + threadsPerBlock.x - 1) / threadsPerBlock.x;
        long double sr = sqrt(blocksPerGrid);

        int gridXY = 0;

          // If square root is an integer
        if((sr - floor(sr)) == 0)
            gridXY = sr;
        else
            gridXY = sr+1;

        numBlocks = dim3(gridXY, gridXY);

    }

    // Kernel invocation
    // dim3 threadsPerBlock(16, 16);
    // dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    printf("CUDA kernel launch with %d %d blocks of %d %d threads\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, type);

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMemcpy(h_C, d_C, N*N*sizeof(h_A[0]), cudaMemcpyDeviceToHost));




    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    
    // Verify that the result vector is correct
    for (int i = 0; i < N*N; ++i)
    {
            if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
            {
                fprintf(stderr, "Result verification failed at element %d!\n", i);
                fprintf(stderr, "we have A[%d] = %f, B[%d] = %f and C[%d] = %f \n", i, h_A[i], i, h_B[i], i,h_C[i]);
                exit(EXIT_FAILURE);
            }
    }

    printf("TEST PASSED\n");


     // Free host memory
     free(h_A);
     free(h_B);
     free(h_C);

    printf("Done\n");
    return 0;
  
}

