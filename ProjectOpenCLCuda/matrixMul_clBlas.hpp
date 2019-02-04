/* ************************************************************************
 * Copyright 2013 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************/


#include <sys/types.h>
#include <stdio.h>
#include <string.h>

/* Include CLBLAS header. It automatically includes needed OpenCL header,
 * so we can drop out explicit inclusion of cl.h header.
 */
// CUDA runtime
#include <cuda_runtime.h>
// #include <cublas_v2.h>

// CUDA and CUBLAS functions
#include <helper_functions.h>
// #include <helper_cuda.h>
#include <clBLAS.h>

// #include "common.h"

void fromFloatTODouble(const float *s, double *d, int size)
{
    for(int i = 0; i< size; i++)
    {
        d[i] = (double) s[i];
    }
}

void fromDoubleTOFloat(const double *s, float *d, int size)
{
    for(int i = 0; i< size; i++)
    {
        d[i] = (double) s[i];
    }
}

////////////////////////////////////////////////////////////////////////////////

 
int matrixMul_clBLAS(const float *h_A, const float *h_B, float *gpu_C, Configuration config)
{

    const clblasOrder order = clblasRowMajor;

    const cl_float alpha = 1.0;

    const clblasTranspose transA = clblasNoTrans;


    const size_t lda = config.WA;        /* i.e. lda = K */

    const clblasTranspose transB = clblasNoTrans;


    const size_t ldb = config.WB;        /* i.e. ldb = N */

    const cl_float beta = 2.0;

    const size_t ldc = config.WB;        /* i.e. ldc = N */





    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem bufA, bufB, bufC;
    cl_event event = NULL;
    int ret = 0;


    printf("Initializing OpenCL device (clBLAS)...\n"); 

    /* Setup OpenCL environment. */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetPlatformIDs() failed with %d\n", err );
        return 1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetDeviceIDs() failed with %d\n", err );
        return 1;
    }

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateContext() failed with %d\n", err );
        return 1;
    }

    queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateCommandQueue() failed with %d\n", err );
        clReleaseContext(ctx);
        return 1;
    }

    /* Setup clblas. */
    err = clblasSetup();
    if (err != CL_SUCCESS) {
        printf("clblasSetup() failed with %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        return 1;
    }

    /* Prepare OpenCL memory objects and place matrices inside them. */
    bufA = clCreateBuffer(ctx, CL_MEM_READ_ONLY, config.size_A * sizeof(float), NULL, &err);
    bufB = clCreateBuffer(ctx, CL_MEM_READ_ONLY, config.size_B  * sizeof(float), NULL, &err);
    bufC = clCreateBuffer(ctx, CL_MEM_READ_WRITE, config.size_C * sizeof(float), NULL, &err);

    err = clEnqueueWriteBuffer(queue, bufA, CL_TRUE, 0, config.size_A * sizeof(float), h_A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufB, CL_TRUE, 0, config.size_B * sizeof(float), h_B, 0, NULL, NULL);

    startTimer();

    /* Call clblas extended function. Perform gemm for the lower right sub-matrices */
    err = clblasSgemm(order, transA, transB, config.HA, config.WB, config.WA,
                         alpha, bufA, 0, lda,
                         bufB, 0, ldb, beta,
                         bufC, 0, ldc,
                         1, &queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSgemmEx() failed with %d\n", err);
        ret = 1;
    }
    else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);


        stopTimer();
        

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0,
                            config.WC * config.HC * sizeof(float),
                            gpu_C, 0, NULL, NULL);

        /* At this point you will get the result of SGEMM placed in 'result' array. */
        puts("");

    }

    /* Release OpenCL events. */
    clReleaseEvent(event);

    /* Release OpenCL memory objects. */
    clReleaseMemObject(bufC);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufA);

    /* Finalize work with clblas. */
    clblasTeardown();

    /* Release OpenCL working objects. */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return ret;
}
