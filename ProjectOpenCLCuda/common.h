#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
struct Configuration
{
    unsigned int WA,HA,WB;
    unsigned int HB,WC,HC;
    unsigned int size_A;
    unsigned int size_B;
    unsigned int size_C;
    unsigned int mem_size_A;
    unsigned int mem_size_B;
    unsigned int mem_size_C;

};

static cudaEvent_t start, stop; 

static void startTimer ()
{
    // Allocate CUDA events that we'll use for timing
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start, NULL);
}

static void stopTimer ()
{
    // Record the stop event
   cudaEventRecord(stop, NULL);

    // Wait for the stop event to complete
   cudaEventSynchronize(stop);

    // Compute and print the performance
    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Kernel execution time on GPU\t: %.5f ms\n", msecTotal);
}

#endif