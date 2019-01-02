#ifndef APP_MANAGER_KERNEL_CUH
#define APP_MANAGER_KERNEL_CUH


#include <stdio.h>
#include <stdlib.h>
#include <helper_cuda.h>



/**
  * CUDA Kernel Device code
  *
  * Computes the vector addition of A and B into C. The 3 vectors have the same
  * number of elements numElements.
  */
__global__ void  vectorAddKernel(const float *A, const float *B, float *C, int numElements, int type)
  {
      int i;
      if (type == 1)
      {
          i = blockIdx.x*blockDim.x+threadIdx.x;
      }
      else
      {
          int blockId = blockIdx.x + blockIdx.y * gridDim.x;
          i = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
      } 

      
      if (i < numElements)
      {
          printf("indice: %d \n",i);   
          C[i] = A[i] + B[i];

          printf("C[%d] %f = %f + %f \n ", i, C[i], A[i], B[i]);
      }
  }

extern "C" void vectorAdd(const float *d_A, const float *d_B, float *d_C, int numElements)
{
    vectorAddKernel<<<16, 16>>>(d_A, d_B, d_C, numElements, 2);  
    getLastCudaError("inverseCNDKernel() execution failed.\n");
}



// __global__ void matrixMultKernel(const float *a,const float *b, float *c, int m, int n, int k)
// { 
//     int row = blockIdx.y * blockDim.y + threadIdx.y; 
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     float sum = 0;

//     // printf("(%d - %d) \n", row,col);

   
//     if( col < k && row < m) 
//     {
//         for(int i = 0; i < n; i++) 
//         {
//             sum += a[row * n + i] * b[i * k + col];

//             // printf("C[%d] %f = %f + %f \n ", i,  sum, a[i], b[i]);
//         }
//         c[row * k + col] = sum;
//         printf("C[%d] %f  \n ", row * k + col,  c[row * k + col]);
//     }
// } 


__global__ void matrixMultKernel(const float *a,const float *b, float *c, int m, int n, int k)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum += a[row * n + i] * b[i * k + col];
            // printf("C[%d] %f = %f + %f \n ", i,  sum, a[i], b[i]);
        }
        c[row * k + col] = sum;
    }
} 


#define BLOCK_SIZE 16
extern "C" void matrixMult(const float *d_A, const float *d_B, float *d_C, int M, int N, int K)
{
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid((M + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);
    matrixMultKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);  
    getLastCudaError("inverseCNDKernel() execution failed.\n");
}

#endif
