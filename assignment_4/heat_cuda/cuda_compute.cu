#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>

static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        printf("cuda error \n");
        exit(1);
    }
}


__global__ void vectorAddKernel(float* deviceA, float* deviceB, float* deviceResult) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
// insert operation here
    deviceResult[i] = deviceA[i]+deviceB[i];
}

extern "C" 
void cuda_do_compute() {
    int threadBlockSize = 512;
    int n=1024; 
    float a[1024], b[1024], result[1024];
    // allocate the vectors on the GPU
    float* deviceA = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceA, n * sizeof(float)));
    if (deviceA == NULL) {
        printf("Error in cudaMalloc! \n");
	return;
    }
    float* deviceB = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceB, n * sizeof(float)));
    if (deviceB == NULL) {
        checkCudaCall(cudaFree(deviceA));
        printf("Error in cudaMalloc! \n");
        return;
    }
    float* deviceResult = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceResult, n * sizeof(float)));
    if (deviceResult == NULL) {
        checkCudaCall(cudaFree(deviceA));
        checkCudaCall(cudaFree(deviceB));
        printf("Error in cudaMalloc! \n");
        return;
    }


    // copy the original vectors to the GPU
    checkCudaCall(cudaMemcpy(deviceA, a, n*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemcpy(deviceB, b, n*sizeof(float), cudaMemcpyHostToDevice));

    // execute kernel
    vectorAddKernel<<<n/threadBlockSize, threadBlockSize>>>(deviceA, deviceB, deviceResult);
    cudaDeviceSynchronize();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    checkCudaCall(cudaMemcpy(result, deviceResult, n * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaCall(cudaMemcpy(b, deviceB, n * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaCall(cudaFree(deviceA));
    checkCudaCall(cudaFree(deviceB));
    checkCudaCall(cudaFree(deviceResult));

}
