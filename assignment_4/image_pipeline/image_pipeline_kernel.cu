#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        printf("cuda Error \n");
	exit(1);
    }
}

__global__ void vectorAddKernel(int* deviceA, int* deviceResult) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
// insert operation here
    deviceResult[i] = deviceA[i];
}
extern "C"
void image_pipeline(int *v, long n){
  int* deviceIn, *deviceOut;
  int threadBlockSize=256;
  int result[256];
  checkCudaCall(cudaMalloc((void **) &deviceIn, n * sizeof(int)));
    if (deviceIn == NULL) {
        printf("Error in cudaMalloc! \n");
        return;
    }
  checkCudaCall(cudaMalloc((void **) &deviceOut, n * sizeof(int)));
    if (deviceOut == NULL) {
        checkCudaCall(cudaFree(deviceIn));
        printf("Error in cudaMalloc! \n");
        return;
    }


    checkCudaCall(cudaMemcpy(deviceIn, v, n * sizeof(int), cudaMemcpyDeviceToHost));
    vectorAddKernel<<<n/threadBlockSize, threadBlockSize>>>(deviceIn, deviceOut);
    cudaDeviceSynchronize();
    checkCudaCall(cudaMemcpy(result, deviceOut, n * sizeof(int), cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(deviceIn));
    checkCudaCall(cudaFree(deviceOut));

}



