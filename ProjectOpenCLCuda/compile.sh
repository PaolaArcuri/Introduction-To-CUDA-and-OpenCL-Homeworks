#!/bin/bash
source ~/setcuda

nvcc -c -I/usr/local/cuda-10.0/include/ -I/usr/local/cuda/samples/common/inc kernel.cu 

if [ ! "$1" == "" ]
then
    nvcc -Xcompiler -I/usr/local/cuda-10.0/include/ -I/usr/local/cuda/samples/common/inc -I ~/libs/clBLAS/usr/local/include/ -L ~/libs/clBLAS/usr/local/lib64/import/  -o $1 kernel.o matrixmul_host.cpp -lOpenCL -lclBLAS -lcublas  -lm --cudart static 
else
    nvcc -Xcompiler -I/usr/local/cuda-10.0/include/ -I/usr/local/cuda/samples/common/inc -I ~/libs/clBLAS/usr/local/include/ -L ~/libs/clBLAS/usr/local/lib64/import/ kernel.o matrixmul_host.cpp -lOpenCL -lclBLAS -lcublas  -lm --cudart static 
fi