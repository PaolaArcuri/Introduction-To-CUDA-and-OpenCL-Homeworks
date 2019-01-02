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
__global__ void  vectorAddKernel(const float *A, const float *B, float *C, int numElements)
  {
      int i;
      if (blockDim.y == 1)
      {
          i = blockIdx.x*blockDim.x+threadIdx.x;
      }
      else if (blockDim.z == 1)
      {
        int blockId = blockIdx.x + blockIdx.y * gridDim.x;
        i = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

        
      }
      else 
      {
        // unique block index inside a 3D block grid
        const unsigned long long int blockId = blockIdx.x //1D
            + blockIdx.y * gridDim.x //2D
            + gridDim.x * gridDim.y * blockIdx.z; //3D

        // global unique thread index, block dimension uses only x-coordinate
        // i = blockId * blockDim.x + threadIdx.x;
        i = blockId * (blockDim.x * blockDim.y *blockDim.z) + ( threadIdx.z *blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;;
      } 

      
      if (i < numElements)
      {
        //   printf("%d  - ",i);   
          C[i] = A[i] + B[i];

        //   printf("C[%d] %f = %f + %f \n ", i, C[i], A[i], B[i]);
      }
  }

extern "C" void vectorAdd(const float *d_A, const float *d_B, float *d_C, int numElements, dim3 threadsPerBlock)
{
    dim3 numBlocks;
    if (threadsPerBlock.y == 1 && threadsPerBlock.z == 1 ) //1D
    {
        numBlocks.x = (numElements + threadsPerBlock.x - 1) / threadsPerBlock.x;
    }
    else if (threadsPerBlock.z == 1) //2D 2D 
    {
        int gridX = (sqrt(numElements) + threadsPerBlock.x - 1) / threadsPerBlock.x;
        int gridY = (sqrt(numElements) + threadsPerBlock.y - 1) / threadsPerBlock.y;
        numBlocks = dim3(gridX, gridY);
    }
    else //3D 3D 
    {
        long double sr = std::pow(numElements, 1/3.);
        int gridX = (sr + threadsPerBlock.x - 1) / threadsPerBlock.x;
        int gridY = (sr + threadsPerBlock.y - 1) / threadsPerBlock.y;
        int gridZ = (sr + threadsPerBlock.z - 1) / threadsPerBlock.z;
        if (sr -floor(sr) != 0 )
        {
            gridX++;
        }
        
        numBlocks = dim3(gridX,gridY,gridZ);
    }

    printf("Thread configuration (%d,%d,%d) \n",threadsPerBlock.x, threadsPerBlock.y,threadsPerBlock.z);
    printf("Block configuration (%d,%d,%d) \n", numBlocks.x, numBlocks.y, numBlocks.z);


    vectorAddKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, numElements);  
    getLastCudaError("vectorAddKernel execution failed.\n");
}




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


extern "C" void matrixMult(const float *d_A, const float *d_B, float *d_C, int M, int N, int K, dim3 threadsPerBlock)
{
    if (threadsPerBlock.y == 1 && threadsPerBlock.z == 1 ) //1D
    {
    }
    else if (threadsPerBlock.z == 1) //2D 2D 
    {
    }
    else 
    {

    }
    
	dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    printf("Thread configuration (%d,%d,%d) \n",threadsPerBlock.x, threadsPerBlock.y,threadsPerBlock.z);
    printf("Block configuration (%d,%d,%d) \n", numBlocks.x, numBlocks.y, numBlocks.z);

    matrixMultKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);  
    getLastCudaError("matrixMult execution failed.\n");
}




// The kernel - DOT PRODUCT
__global__ void dotProductKernel(const float *a,const float *b, float *c) 
{
    extern __shared__ float temp[];

    int index = threadIdx.x + blockIdx.x * blockDim.x;

    temp[threadIdx.x] = a[index] * b[index];
    //Synch threads
    __syncthreads();
    if (0 == threadIdx.x) {
        float sum = 0.00;
        int i;
        for (i=0; i<blockDim.x; i++)
            sum += temp[i];
        atomicAdd(c, sum);        
    } 
}



extern "C" void dotProduct(const float *d_A, const float *d_B, float *d_C, int N, dim3 threadsPerBlock)
{

    dim3 numBlocks;
    if (threadsPerBlock.y == 1 && threadsPerBlock.z == 1 ) //1D
    {
        numBlocks.x = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
    }
    else return;
   
    printf("Thread configuration (%d,%d,%d) \n",threadsPerBlock.x, threadsPerBlock.y,threadsPerBlock.z);
    printf("Block configuration (%d,%d,%d) \n", numBlocks.x, numBlocks.y, numBlocks.z);


    dotProductKernel<<<numBlocks, threadsPerBlock,  threadsPerBlock.x*sizeof(float)>>>(d_A, d_B, d_C);  
    getLastCudaError("dotProduct execution failed.\n");
}




#endif
