// #include "common.h"


// =================================================================================================
// Load an OpenCL kernel from file
char* readKernelFile(const char* filename) {

    // Open the file
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("-- Error opening file %s\n", filename);
        exit(1);
    }

    // Get its size
    fseek(file, 0, SEEK_END);
    long size = ftell(file);
    rewind(file);

    // Read the kernel code as a string
    char* source = (char *)malloc((size+1)*sizeof(char));
    fread(source, 1, size*sizeof(char), file);
    source[size] = '\0';
    fclose(file);

    return source;
}


// =================================================================================================



int setupAndExecuteOpenCLKernel(std::string filename, const float *h_A, const float *h_B, float *gpu_C, Configuration config)
{
   int err;                            // error code returned from api calls

   cl_device_id device_id;             // compute device id 
   cl_context context;                 // compute context
   cl_command_queue commands;          // compute command queue
   cl_program program;                 // compute program
   cl_kernel kernel;                   // compute kernel

   cl_event event_clock;

    // OpenCL device memory for matrices
   cl_mem d_A;
   cl_mem d_B;
   cl_mem d_C;
 
   printf("\nInitializing OpenCL device...\n"); 

   cl_uint dev_cnt = 0;
   clGetPlatformIDs(0, 0, &dev_cnt);
	
   cl_platform_id platform_ids[100];
   clGetPlatformIDs(dev_cnt, platform_ids, NULL);
	
   // Connect to a compute device
   int gpu = 1;
   err = clGetDeviceIDs(platform_ids[0], gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to create a device group!\n");
       return EXIT_FAILURE;
   }
  
   // Create a compute context 
   context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
   if (!context)
   {
       printf("Error: Failed to create a compute context!\n");
       return EXIT_FAILURE;
   }

   // Create a command commands
   commands = clCreateCommandQueue(context, device_id, 0, &err);
   if (!commands)
   {
       printf("Error: Failed to create a command commands!\n");
       return EXIT_FAILURE;
   }

   // Create the compute program from the source file
   char *kernelSource = readKernelFile("matrixmul_kernel.cl");


   program = clCreateProgramWithSource(context, 1, (const char **) & kernelSource, NULL, &err);
   if (!program)
   {
       printf("Error: Failed to create compute program!\n");
       return EXIT_FAILURE;
   }
   // Build the program executable
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if (err != CL_SUCCESS)
   {
       size_t len;
       char buffer[2048];
       printf("Error: Failed to build program executable!\n");
       clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
       printf("%s\n", buffer);
       exit(1);
   }

   // Create the compute kernel in the program we wish to run
   //
   kernel = clCreateKernel(program, "matrixMul", &err);
   if (!kernel || err != CL_SUCCESS)
   {
       printf("Error: Failed to create compute kernel!\n");
       exit(1);
   }

   // Create the input and output arrays in device memory for our calculation
   d_C = clCreateBuffer(context, CL_MEM_READ_WRITE, config.mem_size_A, NULL, &err);
   d_A = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, config.mem_size_A,(void*) h_A, &err);
   d_B = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, config.mem_size_B,(void*) h_B, &err);

   if (!d_A || !d_B || !d_C)
   {
       printf("Error: Failed to allocate device memory!\n");
       exit(1);
   }    
    
   printf("Running matrix multiplication for matrices A (%dx%d) and B (%dx%d) ...\n", config.WA, config.HA,config.WB, config.HB); 

   //Launch OpenCL kernel
   size_t localWorkSize[2], globalWorkSize[2];
 

   err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_C);
   err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_A);
   err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_B);
   err |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&config.WA);
   err |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&config.WC);

   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to set kernel arguments! %d\n", err);
       exit(1);
   }
 
   localWorkSize[0] = 32;
   localWorkSize[1] = 32;
   globalWorkSize[0] = config.WC;
   globalWorkSize[1] = config.HC;


    startTimer();
 
   err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &event_clock);

   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to execute kernel! %d\n", err);
       exit(1);
   }

    err = clWaitForEvents(1, &event_clock);
    if (err != CL_SUCCESS)
    {
       printf("Error: Failed to end event! %d\n", err);
       exit(1);
    }

    stopTimer();

 
   //Retrieve result from device
   err = clEnqueueReadBuffer(commands, d_C, CL_TRUE, 0, config.mem_size_C, gpu_C, 0, NULL, NULL);

   if (err != CL_SUCCESS)
   {
       printf("Error: Failed to read output array! %d\n", err);
       exit(1);
   }

 
   printf("Matrix multiplication completed...\n"); 

   //Shutdown and cleanup
   clReleaseMemObject(d_A);
   clReleaseMemObject(d_C);
   clReleaseMemObject(d_B);

   clReleaseProgram(program);
   clReleaseKernel(kernel);
   clReleaseCommandQueue(commands);
   clReleaseContext(context);

   clReleaseEvent(event_clock);

   return 0;
}