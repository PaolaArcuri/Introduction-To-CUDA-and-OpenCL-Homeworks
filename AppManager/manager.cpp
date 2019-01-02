#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <cassert>

#include "AppConfiguration.h"

using namespace std;

// nvcc -I/usr/local/cuda/samples/common/inc -Xcompiler "-std=c++0x" manager.cpp tinyxml2.cpp -o manager


//nvcc -I/usr/local/cuda/samples/common/inc -c kernels.cu
//nvcc -I/usr/local/cuda/samples/common/inc -Xcompiler "-std=c++0x" manager.cpp tinyxml2.cpp -o manager kernels.o 


// CUDA Runtime
#include <cuda_runtime.h>

// Utilities and system includes
#include <helper_functions.h>
#include <helper_cuda.h>

extern "C" void vectorAdd(const float *A, const float *B, float *C, int numElements, dim3 threadsPerBlock);
extern "C" void matrixMult(const float *d_A, const float *d_B, float *d_C, int N, int M, int K, dim3 threadsPerBlock) ;
extern "C" void dotProduct(const float *d_A, const float *d_B, float *d_C, int N, dim3 threadsPerBlock);
void cpu_matrix_mult(const float *h_A, const float *h_B, float *h_result, int m, int n, int k);
void cpu_vec_add(const float *h_A, const float *h_B, float *h_result, int N);
bool checkSizeInput (const AppConfig appConfig, int & nElements_C );
void cpu_dot_product (const float *h_A, const float *h_B, float *h_result, int N);


int main(int argc, char **argv)
{
  
    assert (argc >1);

     //-----------------------------------------------------------//
     //load confing
    struct AppConfig appConfig;

    try
    {
        loadAppConfig (argv[1], appConfig);

    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error occurred: " << ex.what() << std::endl;
        std::cerr <<"You must specify every element of the xml"<<std::endl;
        return 0;
    }

    printAppConfig(appConfig);
    //-----------------------------------------------------------//

    //-----------------------------------------------------------//
    //Allocating host memory
    float *h_A, *h_B, *h_C, *h_result;


    int nElements_A = appConfig.dim_A_X*appConfig.dim_A_Y;
    int nElements_B = appConfig.dim_B_X*appConfig.dim_B_Y;

    int nElements_C;
     
    size_t size_A = nElements_A * sizeof(float);
    size_t size_B = nElements_B * sizeof(float);


    if(!checkSizeInput(appConfig, nElements_C))
    {
        cerr<<"Invalid Data Size! "<<endl;
        return 0;
    }

    size_t size_C = nElements_C * sizeof(float);
    
    cout<<"num A: "<<nElements_A<<endl;
    cout<<"num B: "<<nElements_B<<endl;
    cout<<"num C: "<<nElements_C<<endl;



    // timing variables
	float gpu_elapsed_time = 0.0;
	cudaEvent_t gpu_start, gpu_stop;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_stop);


    printf("Allocating CPU memory...\n");
    checkCudaErrors(cudaMallocHost((void **) &h_A, size_A));
    checkCudaErrors(cudaMallocHost((void **) &h_B, size_B));
    checkCudaErrors(cudaMallocHost((void **) &h_C, size_C));

    checkCudaErrors(cudaMallocHost((void **) &h_result, size_C));

     //-----------------------------------------------------------//


    for(int i = 0; i < nElements_A; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        
    }
        for(int i = 0; i < nElements_B; i++) {
        h_B[i] = rand() / (float)RAND_MAX;        
    }

    // for(int i = 0; i < nElements_A; i++) {
    //    cout<< h_A[i]<<" ";
    // }

    // cout<<endl;
    // for(int i = 0; i < nElements_B; i++) {
    //    cout<< h_B[i] <<" ";       
    // }
    // cout<<endl;



     //-----------------------------------------------------------//
     printf("Allocating GPU memory...\n");  

    float *d_A, *d_B, *d_C;
    checkCudaErrors(cudaMalloc((void **) &d_A, size_A));
    checkCudaErrors(cudaMalloc((void **) &d_B, size_B));
    checkCudaErrors(cudaMalloc((void **) &d_C, size_C));

    // copy matrix A and B from host to device memory
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
     //-----------------------------------------------------------//

    dim3 threadsPerBlock (appConfig.threads_per_block_X, appConfig.threads_per_block_Y, appConfig.threads_per_block_Z);

    //start timer
    cudaEventRecord(gpu_start, 0);

     //-----------------------------------------------------------//
    if (appConfig.app == APP_TYPE::VECTOR_ADD)
    {
        vectorAdd(d_A, d_B, d_C, nElements_A, threadsPerBlock);

        cout<<"vector add "<<endl;

        cpu_vec_add(h_A,h_B, h_result,nElements_C);
    }

    else if (appConfig.app == APP_TYPE::MATRIX_MUL)
    {
        matrixMult(d_A, d_B, d_C, appConfig.dim_A_X, appConfig.dim_A_Y, appConfig.dim_B_Y, threadsPerBlock);

        cout<<"matrix Multiplication "<<endl;

        cpu_matrix_mult(h_A,h_B, h_result, appConfig.dim_A_X, appConfig.dim_A_Y, appConfig.dim_B_Y);
    }

    else if (appConfig.app == APP_TYPE::DOT_PRODUCT)
    {
       dotProduct(d_A, d_B, d_C, nElements_A, threadsPerBlock);

        cout<<"dot product "<<endl;

        cpu_dot_product(h_A,h_B, h_result,nElements_A);
    }
     //-----------------------------------------------------------//

    //stop timer
    cudaEventRecord(gpu_stop, 0);
	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);

	checkCudaErrors(cudaEventDestroy(gpu_start));
	checkCudaErrors(cudaEventDestroy(gpu_stop));


    printf("Copy output data from the CUDA device to the host memory\n");
    checkCudaErrors(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));


     //-----------------------------------------------------------//
     // Verify that the result is correct
    for (int i = 0; i < nElements_C; ++i)
    {
        printf ("%0.5f %0.5f \n", h_C[i], h_result[i]);
        // cout<<" i = "<<i <<"   " << h_C[i]<< " h result " << h_result[i] << " fabs(h_result[i] - h_C[i])  " << fabs(h_result[i] - h_C[i]) <<endl;
        if (fabs(h_result[i] - h_C[i]) > 1e-6)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    //-----------------------------------------------------------//

    printf("Test PASSED\n");

    printf("Execution Time: %fms\n", gpu_elapsed_time);



    // Free host memory
    checkCudaErrors(cudaFreeHost(h_A));
    checkCudaErrors(cudaFreeHost(h_B));
    checkCudaErrors(cudaFreeHost(h_C));
     

    //Free device memory
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));







    return 0;
}


void cpu_matrix_mult(const float *h_A, const float *h_B, float *h_result, int m, int n, int k) {
    for (int i = 0; i < m; ++i) 
    {
        for (int j = 0; j < k; ++j) 
        {
            float tmp = 0.0;
            for (int h = 0; h < n; ++h) 
            {
                tmp += h_A[i * n + h] * h_B[h * k + j];
            }
            h_result[i * k + j] = tmp;

        }
    }

}

void cpu_vec_add(const float *h_A, const float *h_B, float *h_result, int N) {
    for (int i = 0; i < N; ++i) 
    {
        h_result[i] = h_A[i] + h_B[i];
    }
}

void cpu_dot_product (const float *h_A, const float *h_B, float *h_result, int N)
{
    h_result[0]=0.0f;
    for (int i=0;i<N;++i) {
        h_result[0]+=h_A[i]*h_B[i];
    }

    printf("%0.6f\n", h_result[0]);
    cout<<h_result[0]<<endl;
}



bool checkSizeInput (const AppConfig appConfig, int & nElements_C )
{
    if (appConfig.app == APP_TYPE::VECTOR_ADD)
    {
        nElements_C = appConfig.dim_A_X;
        return ((appConfig.dim_A_X == appConfig.dim_B_X) && appConfig.dim_A_Y == 1 && appConfig.dim_B_Y == 1);
    }
    if (appConfig.app == APP_TYPE::MATRIX_MUL)
    { 
        nElements_C = appConfig.dim_A_X * appConfig.dim_B_Y;
        return (appConfig.dim_A_Y == appConfig.dim_B_X);
    }

    if (appConfig.app == APP_TYPE::DOT_PRODUCT) 
    { 
        nElements_C = 1;
        return ((appConfig.dim_A_X == appConfig.dim_B_X) && appConfig.dim_A_Y == 1 && appConfig.dim_B_Y == 1);
    }
}


