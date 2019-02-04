////////////////////////////////////////////////////////////////////////////////

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include <stdbool.h>
#include <iostream>

#include "common.h"

#include "matrixMul_clBlas.hpp"

#include "matrixMul_CUBLAS.hpp"

#include "matrixmul_opencl.hpp"


// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>





extern "C" void MatrixMultiply(float* h_A, float* h_B, float *gpu_C, int block_size, Configuration configuration);

//nvcc -c -I/usr/local/cuda-10.0/include/ -I/usr/local/cuda/samples/common/inc  kernel.cu 
// nvcc -Xcompiler -I/usr/local/cuda-10.0/include/ -I/usr/local/cuda/samples/common/inc -I ~/libs/clBLAS/usr/local/include/ -L ~/libs/clBLAS/usr/local/lib64/import/  -o m2 kernel.o matrixmul_host.cpp -lOpenCL -lclBLAS -lcublas  -lm --cudart static 



////////////////////////////////////////////////////////////////////////////////

 
// Allocates a matrix with random float entries.
void randomMemInit(float* data, int size)
{
   int i;

   for (i = 0; i < size; ++i)
   	data[i] = rand() / (float)RAND_MAX;
}

// =================================================================================================
// get a matrix (NxM) and produce its transposition
void transpose(float *src, float *dst, const int N, const int M, Configuration config) {
    // #pragma omp parallel for
     for(int i = 0; i < config.HC; i++)
    {
        for(int j=0; j< config.WC; j++)
        {
            dst[i*config.WC+j] = src[j*config.HC+i];
        }
    }
}

// =================================================================================================

void matrixMulCPU(float *C, const float *A, const float *B, const Configuration config)
{
    unsigned int i,j,k;
    for (i = 0; i < config.HA; ++i)
        for (j = 0; j < config.WB; ++j)
        {
            double sum = 0;

            for (k = 0; k < config.WA; ++k)
            {
                double a = A[i * config.WA + k];
                double b = B[k * config.WB + j];
                sum += a * b;
            }

            C[i * config.WB + j] = (float)sum;
        }
}

//compare and count how many elements are different between two matrices
bool compare (const float* A, const float* B, const Configuration config)
{
    double delta;
    int i, c = 0;
    for(i = 0; i < config.size_C; i++)
   {

        delta = fabs(A[i] - B[i]);
        if(delta > 1e-5)
        {
            c++;
            printf("gpu %.6f cpu %.6f and diff is %.6f \n", A[i], B[i], delta);    
        }
   }
   printf("count = %d \n", c);
   return (c == 0);
}

//compare a matrix to the CPU result
bool compareResultWithCPU (float* h_A, float* h_B, float* h_C, const Configuration config)
{
    float *cpu_result = (float*) malloc(config.mem_size_C);
    matrixMulCPU (cpu_result, h_A,h_B, config);

    return compare (h_C, cpu_result, config);

}


//check if the matrix is symmetric
bool symmetricCheck (float * h_C, const Configuration config)
{
    for(int i = 0; i < config.HC; i++)
    {
        for(int j=0; j< config.WC; j++)
        {
            if( (h_C[i*config.WC+j] != h_C[j*config.HC+i]))
            {
                return false;
            }
        }
    }

    return true;
}

int main(int argc, char **argv)
{

    Configuration config;
    config.WA = strtol(argv[1], NULL, 10);
    // config.HA = strtol(argv[2], NULL, 10);
    // config.WB = strtol(argv[3], NULL, 10);
    config.HA = config.WA;
    config.WB = config.WA;

    //defining matrix comfiguration
    config.HB = config.WA;
    config.WC = config.WB;
    config.HC = config.HA;

    config.size_A = config.WA * config.HA;
    config.mem_size_A = sizeof(float) * config.size_A;

    config.size_B = config.WB * config.HB;
    config.mem_size_B = sizeof(float) * config.size_B;

    config.size_C = config.WC * config.HC;
    config.mem_size_C = sizeof(float) * config.size_C;

 
    //Allocate host memory for matrices A and B
   float* h_A = (float*) malloc(config.mem_size_A);
 
   
   float* h_B = (float*) malloc(config.mem_size_B);


   //Initialize host memory
   randomMemInit(h_A, config.size_A);
   transpose(h_A, h_B, config.WA, config.WB, config);

   //OPENCL VERSION
   float* h_C = (float*) malloc(config.mem_size_C);

   setupAndExecuteOpenCLKernel("matrixmul_kernel.cl",h_A, h_B, h_C, config);

    if (symmetricCheck(h_C, config))
        printf("OPENCL Test Passed \n-------------------------------------------------------------\n\n");
        
    else
    {
         printf("OPENCL Test FAILED \n-------------------------------------------------------------\n\n");
    }
    
     //CUDA VERSION
    float* cuda_C = (float*) malloc(config.mem_size_C);
    MatrixMultiply(h_A,h_B,cuda_C, 32, config);

    if (symmetricCheck(cuda_C, config))
        printf("cuda Test Passed \n-------------------------------------------------------------\n\n");
        
    else
    {
         printf("cuda Test FAILED \n-------------------------------------------------------------\n\n");
    }

    //CLBLAS VERSION
    float* clBLAS_C = (float*) malloc(config.mem_size_C);

    matrixMul_clBLAS(h_A, h_B, clBLAS_C, config);

    if (symmetricCheck(clBLAS_C, config))
        printf("clBLAS Test Passed \n-------------------------------------------------------------\n\n");
        
    else
    {
         printf("clBLAS Test FAILED \n-------------------------------------------------------------\n\n");
    }

    
    //CUBLAS VERSION
    int devID = 0;
    float* cuBLAS_C = (float*) malloc(config.mem_size_C);
    initializeCUDA(devID, config);

    matrixMultiply_CUBLAS(h_A, h_B, cuBLAS_C, devID, config );

    if (symmetricCheck(clBLAS_C, config))
        printf("cuBLAS Test Passed \n-------------------------------------------------------------\n\n");
        
    else
    {
         printf("cuBLAS Test FAILED \n-------------------------------------------------------------\n\n");
    }



//    compareResultWithCPU(h_A,h_B,cuBLAS_C,config);

    // float* cpu_result = (float*) malloc(config.mem_size_C);

    // matrixMulCPU(cpu_result, h_A, h_B, config);

//    compareResultWithCPU(h_A,h_B,clBLAS_C,config);
    if(compare(h_C, cuda_C, config))
    {
        printf("CUDA and OpenCL produced the same result \n-------------------------------------------------------------\n\n");
    }

    if(compare(cuBLAS_C, clBLAS_C, config))
    {
        printf("cuBLAS and clBLAS produced the same result \n-------------------------------------------------------------\n\n");
    }

    //Shutdown and cleanup
    free(h_A);
    free(h_B);

    // free(cpu_result);


    free(h_C);
    free(clBLAS_C);
    free(cuBLAS_C);
    free(cuda_C);

}
