## APP MANAGER

the library **TinyXML-2** has been used to read and parse the xml file that contains app configurations! 

<https://github.com/leethomason/tinyxml2>

compilation and execution

```bash 
nvcc -I/usr/local/cuda/samples/common/inc -c kernels.cu
nvcc -I/usr/local/cuda/samples/common/inc -Xcompiler "-std=c++0x" manager.cpp tinyxml2.cpp -o manager kernels.o 
 ./manager configuration.xml
```
