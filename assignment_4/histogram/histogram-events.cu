#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "timer.h"

using namespace std;

/* Utility function, use to do error checking.

   Use this function like this:

   checkCudaCall(cudaMalloc((void **) &deviceRGB, imgS * sizeof(color_t)));

   And to check the result of a kernel invocation:

   checkCudaCall(cudaGetLastError());
*/
static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        cerr << "cuda error: " << cudaGetErrorString(result) << endl;
        exit(1);
    }
}


__global__ void histogramKernel(unsigned char* image, long img_size, unsigned int* histogram, int hist_size) {
// insert operation here

}

void histogramCuda(unsigned char* image, long img_size, unsigned int* histogram, int hist_size) {
    int threadBlockSize = 512;
    float kernelTime = 0;
    float h2dTime, d2hTime, memTime = 0;
    cudaEvent_t s1,s2,s3,s4,s5,s6;

    // allocate the vectors on the GPU
    unsigned char* deviceImage = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceImage, img_size * sizeof(unsigned char)));
    if (deviceImage == NULL) {
        cout << "could not allocate memory!" << endl;
        return;
    }
    unsigned int* deviceHisto = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceHisto, hist_size * sizeof(unsigned int)));
    if (deviceHisto == NULL) {
        checkCudaCall(cudaFree(deviceImage));
        cout << "could not allocate memory!" << endl;
        return;
    }
    cudaEventCreate(&s1);
    cudaEventCreate(&s2);
    cudaEventCreate(&s3);
    cudaEventCreate(&s4);
    cudaEventCreate(&s5);
    cudaEventCreate(&s6);

    // copy the original vectors to the GPU
    cudaEventRecord(s1,0);
    checkCudaCall(cudaMemcpy(deviceImage, image, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice));
    cudaEventRecord(s2,0);
    
    // execute kernel
    cudaEventRecord(s3,0);
    histogramKernel<<<img_size/threadBlockSize, threadBlockSize>>>(deviceImage, img_size, deviceHisto, hist_size);
    cudaEventRecord(s4,0);

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    cudaEventRecord(s5,0);
    checkCudaCall(cudaMemcpy(histogram, deviceHisto, hist_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaEventRecord(s6,0);

    checkCudaCall(cudaFree(deviceImage));
    checkCudaCall(cudaFree(deviceHisto));

    cudaEventSynchronize(s6);

    cudaEventElapsedTime(&h2dTime, s1, s2);
    cudaEventElapsedTime(&kernelTime, s3, s4);
    cudaEventElapsedTime(&d2hTime, s5, s6);

    cout << "histogram (kernel): \t\t" << kernelTime / 1000 << " seconds."  << endl;
    cout << "histogram (memory): \t\t" << (h2dTime+d2hTime)/1000 << " seconds."  << endl;
/*
   cudaEventDestroy(s1);
   cudaEventDestroy(s2);
   cudaEventDestroy(s3);
   cudaEventDestroy(s4);
    cudaEventDestroy(s5);
   cudaEventDestroy(s6);
*/
}

void histogramSeq(unsigned char* image, long img_size, unsigned int* histogram, int hist_size) {
  int i; 

  timer sequentialTime = timer("Sequential");
  
  for (i=0; i<hist_size; i++) histogram[i]=0;

  sequentialTime.start();
  for (i=0; i<img_size; i++) {
	histogram[image[i]]++;
  }
  sequentialTime.stop();
  
  cout << "histogram (sequential): \t\t" << sequentialTime << endl;

}

int main(int argc, char* argv[]) {
    long img_size = 655360;
    int hist_size = 256;
    
    if (argc > 1) img_size = atoi(argv[1]);
    if (img_size < 1024) {
	cout << "Error in parameter" << endl;
	exit(-1);
    }

    unsigned char *image = (unsigned char *)malloc(img_size * sizeof(unsigned char)); 
    unsigned int *histogramS = (unsigned int *)malloc(hist_size * sizeof(unsigned int));     
    unsigned int *histogram = (unsigned int *)malloc(hist_size * sizeof(unsigned int));

    // initialize the vectors.
    for(long i=0; i<img_size; i++) {
        image[i] = (unsigned char) (i % hist_size);
    }

    cout << "Compute the histogram of a gray image with " << img_size << " pixels." << endl;

    histogramSeq(image, img_size, histogramS, hist_size);
    histogramCuda(image, img_size, histogram, hist_size);
    
    // verify the resuls
    for(int i=0; i<hist_size; i++) {
	  if (histogram[i]!=histogramS[i]) {
            cout << "error in results! Bin " << i << " is "<< histogram[i] << ", but should be " << histogramS[i] << endl; 
            exit(1);
        }
    }
    cout << "results OK!" << endl;
     
    free(image);
    free(histogram);
    free(histogramS);         
    
    return 0;
}
