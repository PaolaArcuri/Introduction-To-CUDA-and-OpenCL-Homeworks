////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//
// Matrix multiplication: C = A * B.
// Host code.
//
// This sample implements matrix multiplication as described in Chapter 3
// of the programming guide and uses the CUBLAS library to demonstrate
// the best performance.

// SOME PRECAUTIONS:
// IF WE WANT TO CALCULATE ROW-MAJOR MATRIX MULTIPLY C = A * B,
// WE JUST NEED CALL CUBLAS API IN A REVERSE ORDER: cublasSegemm(B, A)!
// The reason is explained as follows:

// CUBLAS library uses column-major storage, but C/C++ use row-major storage.
// When passing the matrix pointer to CUBLAS, the memory layout alters from
// row-major to column-major, which is equivalent to an implicit transpose.

// In the case of row-major C/C++ matrix A, B, and a simple matrix multiplication
// C = A * B, we can't use the input order like cublasSgemm(A, B)  because of
// implicit transpose. The actual result of cublasSegemm(A, B) is A(T) * B(T).
// If col(A(T)) != row(B(T)), equal to row(A) != col(B), A(T) and B(T) are not
// multipliable. Moreover, even if A(T) and B(T) are multipliable, the result C
// is a column-based cublas matrix, which means C(T) in C/C++, we need extra
// transpose code to convert it to a row-based C/C++ matrix.

// To solve the problem, let's consider our desired result C, a row-major matrix.
// In cublas format, it is C(T) actually (because of the implicit transpose).
// C = A * B, so C(T) = (A * B) (T) = B(T) * A(T). Cublas matrice B(T) and A(T)
// happen to be C/C++ matrice B and A (still because of the implicit transpose)!
// We don't need extra transpose code, we only need alter the input order!
//
// CUBLAS provides high-performance matrix multiplication.
// See also:
// V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
// in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
// Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
//

// [cuda-s12@lhcbgpu1 src]$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
// [cuda-s12@lhcbgpu1 src]$ nvcc -I"/usr/local/cuda-10.0/samples/0_Simple" -I"/usr/local/cuda-10.0/samples/common/inc" -G -g -O0 --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61 -lcublas  cublas.cpp -o m1

// Utilities and system includes
#include <assert.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

// CUDA runtime
// #include <cuda_runtime.h>
#include <cublas_v2.h>

#include <helper_functions.h>

// CUDA and CUBLAS functions
// #include <helper_functions.h>
#include <helper_cuda.h>

// #include "common.h"



void initializeCUDA(int &devID, Configuration config)
{
    // By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
    cudaError_t error;
    devID = 0;

    cudaDeviceProp deviceProp;

    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess)
    {
        printf("cudaGetDeviceProperties returned error code %d, line(%d)\n", error, __LINE__);
        exit(EXIT_FAILURE);
    }

    printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);

    int block_size = 32;


    printf("MatrixA(%u,%u), MatrixB(%u,%u), MatrixC(%u,%u)\n",
           config.HA, config.WA,
           config.HB, config.WB,
           config.HC, config.WC);

    if( config.WA != config.HB ||
        config.HA != config.HC ||
        config.WB != config.WC)
    {
       printf("ERROR: Matrix sizes do not match!\n");
       exit(-1);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Run a test matrix multiply using CUBLAS
////////////////////////////////////////////////////////////////////////////////
int matrixMultiply_CUBLAS(const float *h_A, const float *h_B, float *h_CUBLAS, int devID, Configuration config)
{
    cudaDeviceProp deviceProp;

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    int block_size = 32;

    // allocate device memory
    float *d_A, *d_B, *d_C;
    

    checkCudaErrors(cudaMalloc((void **) &d_A, config.mem_size_A));
    checkCudaErrors(cudaMalloc((void **) &d_B, config.mem_size_B));
    checkCudaErrors(cudaMemcpy(d_A, h_A, config.mem_size_A, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, config.mem_size_B, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void **) &d_C, config.mem_size_C));

    // setup execution parameters
    dim3 threads(block_size, block_size);
    dim3 grid(config.WC / threads.x, config.HC / threads.y);

    // create and start timer
    printf("Computing result using CUBLAS...");

    // execute the kernel
    const float alpha = 1.0f;
    const float beta  = 2.0f;
    cublasHandle_t handle;
    

    checkCudaErrors(cublasCreate(&handle));
        
   
    startTimer();

    checkCudaErrors(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, config.WB, config.HA, config.WA, &alpha, d_B, config.WB, d_A, config.WA, &beta, d_C, config.WB));

    printf("done.\n");

    stopTimer();
       
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_CUBLAS, d_C, config.mem_size_C, cudaMemcpyDeviceToHost));

    // Destroy the handle
    checkCudaErrors(cublasDestroy(handle));


    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));

}


