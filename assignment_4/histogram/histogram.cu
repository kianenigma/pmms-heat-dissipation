#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include "timer.h"

using namespace std;

#define hist_size 256

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


__global__ void histogramKernel(unsigned char* __restrict__ image, long img_size, unsigned int* __restrict__ histos) {
    __shared__ unsigned int shared_histo[hist_size];
    unsigned int tid = threadIdx.x;
    unsigned int i = tid + blockDim.x * blockIdx.x;

    // initialize shared memory to 0 in parallel (256 first threads in each block)
    if(tid < hist_size) {
        shared_histo[tid] = 0;
    }
    // make sure, that all writes to shared memory are finished
    __syncthreads();

    if(i < img_size) {
        atomicAdd(&shared_histo[image[i]], 1);
    }
    // make sure, that all writes to shared memory are finished
    __syncthreads();

    // write histogram of block back to global memory
    if(tid < hist_size) {
        // advance pointer to histograms to block specific one
        histos += blockIdx.x * hist_size;
        histos[tid] = shared_histo[tid];
    }
}

__global__ void reduceKernel(unsigned int* __restrict__ histos, const int reduce_blocks, const int last_reduction_blocks) {
    unsigned int tid = threadIdx.x;
    
    if((blockIdx.x + reduce_blocks) < last_reduction_blocks) {
        // get current position
        int thread = blockIdx.x * hist_size + tid;

        // get position from block to reduce
        int thread_next = (blockIdx.x + reduce_blocks) * hist_size + tid;

        histos[thread] += histos[thread_next];
    }
}

void histogramCuda(unsigned char* image, long img_size, unsigned int* histogram) {
    int threadBlockSize = 1024;
    int blocks;

    // calculate number of blocks based on img_size
    blocks = img_size / threadBlockSize;
    if(img_size % threadBlockSize != 0) {
        blocks++;
    }

    // allocate the vectors on the GPU
    unsigned char* deviceImage = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceImage, img_size * sizeof(unsigned char)));
    if (deviceImage == NULL) {
        cout << "could not allocate memory!" << endl;
        return;
    }
    unsigned int* deviceHistos = NULL;
    checkCudaCall(cudaMalloc((void **) &deviceHistos, blocks * hist_size * sizeof(unsigned int)));
    if (deviceHistos == NULL) {
        checkCudaCall(cudaFree(deviceImage));
        cout << "could not allocate memory!" << endl;
        return;
    }

    timer kernelTime1 = timer("kernelTime1");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceImage, image, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice));
    memoryTime.stop();

    // execute kernel
    kernelTime1.start();
    histogramKernel<<<blocks, threadBlockSize>>>(deviceImage, img_size, deviceHistos);
    cudaDeviceSynchronize();
    kernelTime1.stop();
    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    int reduce_blocks = (int) ceil(blocks / 2.0);
    if(reduce_blocks % 2 != 0) {
        reduce_blocks++;
    }
    int last_reduction = blocks;
    
    while(reduce_blocks >= 1) {
        // execute reduce kernel
        kernelTime1.start();
        reduceKernel<<<reduce_blocks, hist_size>>>(deviceHistos, reduce_blocks, last_reduction);
        cudaDeviceSynchronize();
        kernelTime1.stop();
        // check whether the kernel invocation was successful
        checkCudaCall(cudaGetLastError());

        if(floor(reduce_blocks / 2.0) == 0) {
            break;
        }

        last_reduction = reduce_blocks;
        reduce_blocks = (int) ceil(reduce_blocks / 2.0);
        if(reduce_blocks % 2 != 0 && reduce_blocks != 1) {
            reduce_blocks++;
        }
    }

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(histogram, deviceHistos, hist_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceImage));
    checkCudaCall(cudaFree(deviceHistos));

    cout << "histogram (kernel): \t\t" << kernelTime1  << endl;
    cout << "histogram (memory): \t\t" << memoryTime << endl;
    cout << "histogram total: \t\t  = " << (kernelTime1.getTimeInSeconds() + memoryTime.getTimeInSeconds()) << " seconds" << endl;
}


__global__ void histogramKernelSimple(unsigned char* __restrict__ image, long img_size, unsigned int* __restrict__ histogram) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if(i < img_size) {
        atomicAdd(&histogram[image[i]], 1);
    }
}

void histogramCudaSimple(unsigned char* image, long img_size, unsigned int* histogram) {
    int threadBlockSize = 1024;
    int blocks;

    // calculate number of blocks based on img_size
    blocks = img_size / threadBlockSize;
    if(img_size % threadBlockSize != 0) {
        blocks++;
    }

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

    timer kernelTime1 = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // copy the original vectors to the GPU
    memoryTime.start();
    checkCudaCall(cudaMemcpy(deviceImage, image, img_size*sizeof(unsigned char), cudaMemcpyHostToDevice));
    checkCudaCall(cudaMemset(deviceHisto, 0, hist_size * sizeof(unsigned int)));
    memoryTime.stop();

    // execute kernel
    kernelTime1.start();
    histogramKernelSimple<<<blocks, threadBlockSize>>>(deviceImage, img_size, deviceHisto);
    cudaDeviceSynchronize();
    kernelTime1.stop();

    // check whether the kernel invocation was successful
    checkCudaCall(cudaGetLastError());

    // copy result back
    memoryTime.start();
    checkCudaCall(cudaMemcpy(histogram, deviceHisto, hist_size * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    memoryTime.stop();

    checkCudaCall(cudaFree(deviceImage));
    checkCudaCall(cudaFree(deviceHisto));

    cout << "histogram simple (kernel): \t" << kernelTime1  << endl;
    cout << "histogram simple (memory): \t" << memoryTime << endl;
    cout << "histogram simple total: \t  = " << (kernelTime1.getTimeInSeconds() + memoryTime.getTimeInSeconds()) << " seconds" << endl;
}

void histogramSeq(unsigned char* image, long img_size, unsigned int* histogram) {
  int i; 

  timer sequentialTime = timer("Sequential");
  
  for (i=0; i<hist_size; i++) histogram[i]=0;

  sequentialTime.start();
  for (i=0; i<img_size; i++) {
	histogram[image[i]]++;
  }
  sequentialTime.stop();
  
  cout << "histogram (seq): \t\t" << sequentialTime << endl;
}


/*
    old: make clean && make && prun -v -1 -np 1 -native '-l gpu=GTX480' ./myhistogram
    new: make clean && make && prun -v -1 -np 1 -native '-C GTX480 --gres=gpu:1' ./myhistogram
    -s executes simple histogram kernel, default=advanced kernel
    -l size of histgram image, default=655360
*/
int main(int argc, char* argv[]) {
    int c;
    long img_size = 655360;
    int simple = 0;


    while((c = getopt(argc, argv, "l:s")) != -1) {
        switch(c) {
            case 'l':
                img_size = atoi(optarg);
                break;
            case 's':
                simple = 1;
                break;
            case '?':
                if(optopt == 'l') {
                    fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                }
                else if(isprint(optopt)) {
                    fprintf(stderr, "Unknown option '-%c'.\n", optopt);
                }
                else {
                    fprintf(stderr, "Unknown option character '\\x%x'.\n", optopt);
                }
                return -1;
            default:
                return -1;
        }
    }

    unsigned char *image = (unsigned char *)malloc(img_size * sizeof(unsigned char)); 
    unsigned int *histogramS = (unsigned int *)malloc(hist_size * sizeof(unsigned int));     
    unsigned int *histogram = (unsigned int *)malloc(hist_size * sizeof(unsigned int));

    // initialize the vectors.
    for(long i=0; i<img_size; i++) {
        //image[i] = (unsigned char) (rand() % hist_size);
        image[i] = 0;
    }

    cout << "Compute the histogram of a gray image with " << img_size << " pixels." << endl;

    histogramSeq(image, img_size, histogramS);

    if(simple == 1) {
        histogramCudaSimple(image, img_size, histogram);
    } else {
        histogramCuda(image, img_size, histogram);  
    }

    // verify the results
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
