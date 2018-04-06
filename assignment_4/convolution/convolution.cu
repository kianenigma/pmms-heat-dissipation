#include <stdio.h>
#include <string.h>
#include "timer.h"
#include "getopt.h"

#define image_height 10000
#define image_width 10000
#define filter_height 5
#define filter_width 5

#define border_height ((filter_height/2)*2)
#define border_width ((filter_width/2)*2)
#define input_height (image_height + border_height)
#define input_width (image_width + border_width)

#define block_size_x 32
#define block_size_y 32

#define SEED 1234

using namespace std;

__constant__ float dc_filter[filter_width*filter_height]; 
__constant__ float dc_filter_sum = 0.0; 

void convolutionSeq(float *output, float *input, float *filter, float filter_sum) {
  timer sequentialTime = timer("Sequential");
  
  sequentialTime.start();

  for (int y=0; y < image_height; y++) {
    for (int x=0; x < image_width; x++) { 
        float newval = 0.0; 
        //for each filter weight
        for (int i=0; i < filter_height; i++) {
            for (int j=0; j < filter_width; j++) {
                newval += input[(y+i)*input_width+x+j] * filter[i*filter_width+j];
            }
        }
        output[y*image_width+x] = newval / filter_sum; 
    }
}
sequentialTime.stop(); 
cout << "convolution (sequential): \t\t" << sequentialTime << endl;

}

/**
    Convolution kernel with constant memory of filter.
*/
__global__ void convolution_kernel(float *output, float *input) {
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; 
    float sum = 0.0;

    if (x >= image_width || y >= image_height ) {
        return;
    }

    for (int i=0; i < filter_height; i++) {
        for (int j=0; j < filter_width; j++) {
            sum += input[(y+i)*input_width+x+j] * dc_filter[i*filter_width+j];
        }
    }
    output[y*image_width+x] = sum / dc_filter_sum; 
}

/**
    Naive convolution kernel with no further optimization.
*/
__global__ void convolution_kernel_naive(float *output, float *input, float *filter, float filter_sum) {
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; 

    float sum = 0.0;

    if (x >= image_width || y >= image_height ) {
        return;
    }

    for (int i=0; i < filter_height; i++) {
        for (int j=0; j < filter_width; j++) {
            sum += input[(y+i)*input_width+x+j] * filter[i*filter_width+j];
        }
    }
    output[y*image_width+x] = sum / filter_sum; 
}

/**
    Convolution kernel with constant memory of filter shared memory usage of inputs per block.
    Results as expected but slow performance due to one thread working all other threads in a block idling.
*/
__global__ void convolution_kernel_shared_mem_naive(float *output, float *input) {
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b_x_start = blockIdx.x * blockDim.x;
    const unsigned int b_y_start = blockIdx.y * blockDim.y;
    const unsigned int t_y = threadIdx.y;
    const unsigned int t_x = threadIdx.x;
    float sum = 0.0;
    const unsigned int s_height = block_size_y + border_height;
    const unsigned int s_width = block_size_x + border_width;
    __shared__ float s_input[s_height * s_width];

    if (x >= image_width || y >= image_height ) {
        return;
    }

    // only first thread in each block
    if(t_y == 0 && t_x == 0) {
        for (int i=0; i < s_height; i++) {
            for (int j=0; j < s_width; j++) {
                s_input[i*s_width+j] = input[(i+b_y_start)*input_width+j+b_x_start];
            }
        }
    }
    __syncthreads();

    for (int i=0; i < filter_height; i++) {
        for (int j=0; j < filter_width; j++) {
            sum += s_input[(t_y+i)*s_width+t_x+j] * dc_filter[i*filter_width+j];
        }
    }

    output[y*image_width+x] = sum / dc_filter_sum;
}

/**
    Convolution kernel with constant memory of filter shared memory usage of inputs per block.
    Results were not as expected when copying values from global memory to shared memory in parallel. 
*/
__global__ void convolution_kernel_shared_mem_parallel_try(float *output, float *input) {
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int b_x_start = blockIdx.x * blockDim.x;
    const unsigned int b_y_start = blockIdx.y * blockDim.y;
    const unsigned int t_y = threadIdx.y;
    const unsigned int t_x = threadIdx.x;
    float sum = 0.0;

    if (x >= image_width || y >= image_height ) {
        return;
    }

    const unsigned int s_height = block_size_y + border_height;
    const unsigned int s_width = block_size_x + border_width;
    __shared__ float s_input[s_height * s_width];

    // copy values 1 to 1 from global memory to shared memory in parallel
    s_input[t_y*s_width+t_x] = input[y*input_width+x];

    // copy the 4 extra columns for every row
    if(t_x == 0) {
        for (int i=block_size_x; i < s_width; i++) {
            s_input[t_y*s_width+i] = input[y*input_width+x+i];
        }
    }

    // copy the 4 extra rows
    if(t_x == 1 && t_y < 4) {
        const unsigned int row = t_y+block_size_y;
        const unsigned int row_global = y+block_size_y;

        for (int i=0; i < s_width; i++) {
            s_input[row*s_width+i] = input[row_global*input_width+i+b_x_start];
        }
    }
    __syncthreads();

    for (int i=0; i < filter_height; i++) {
        for (int j=0; j < filter_width; j++) {
            sum += s_input[(t_y+i)*s_width+t_x+j] * dc_filter[i*filter_width+j];
        }
    }

    output[y*image_width+x] = sum / dc_filter_sum;
}

void convolutionCUDA(float *output, float *input, float *filter, float filter_sum) {   
    float *d_input; float *d_output; 

    cudaError_t err;
    timer kernelTime = timer("kernelTime");
    timer memoryTime = timer("memoryTime");

    // memory allocation
    err = cudaMalloc((void **)&d_input, input_height*input_width*sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMalloc d_input: [%d] %s\n",err, cudaGetErrorString( err )); }
    err = cudaMalloc((void **)&d_output, image_height*image_width*sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMalloc d_output: %s\n", cudaGetErrorString( err )); }

    memoryTime.start();
    // host to device 
    err = cudaMemcpy(d_input, input, input_height*input_width*sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMemcpy host to device input: %s\n", cudaGetErrorString( err ));  }
    
    // Copy constant memory data
    err = cudaMemcpyToSymbol(dc_filter_sum, &filter_sum, sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMemcpyToSymbol output: %s\n", cudaGetErrorString( err ));  }
    err = cudaMemcpyToSymbol(dc_filter, filter, filter_width*filter_height*sizeof(float));
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMemcpyToSymbol output: %s\n", cudaGetErrorString( err ));  }

    memoryTime.stop();

    //setup the grid and thread blocks
    //thread block size
    dim3 threads(block_size_x, block_size_y);
    //problem size divided by thread block size rounded up
    dim3 grid(int(ceilf(image_width/(float)threads.x)), int(ceilf(image_height/(float)threads.y)) );

    printf("image [%d %d] | input [%d %d]\n", image_height, image_width, input_height, input_width);
    printf("GRID DIM [%d %d] | BLOCK DIM [%d %d]\n", grid.y, grid.x, threads.y, threads.x);

    //measure the GPU function
    kernelTime.start();
    convolution_kernel<<<grid, threads>>>(d_output, d_input);
    cudaDeviceSynchronize();
    kernelTime.stop();

    //check to see if all went well
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "Error during kernel launch convolution_kernel: %s\n", cudaGetErrorString( err )); }

    //copy the result back to host memory
    memoryTime.start();
    err = cudaMemcpy(output, d_output, image_height*image_width*sizeof(float), cudaMemcpyDeviceToHost);
    memoryTime.stop();
    if (err != cudaSuccess) { fprintf(stderr, "Error in cudaMemcpy device to host output: %s\n", cudaGetErrorString( err )); }

    err = cudaFree(d_input);
    if (err != cudaSuccess) { fprintf(stderr, "Error in freeing d_input: %s\n", cudaGetErrorString( err )); }
    err = cudaFree(d_output);
    if (err != cudaSuccess) { fprintf(stderr, "Error in freeing d_output: %s\n", cudaGetErrorString( err )); }
    // err = cudaFree(d_filter);
    // if (err != cudaSuccess) { fprintf(stderr, "Error in freeing d_filter: %s\n", cudaGetErrorString( err )); }

    cout << "convolution (kernel): \t\t" << kernelTime << endl;
    cout << "convolution (memory): \t\t" << memoryTime << endl;

}

int compare_arrays(float *a1, float *a2, int n) {
    int errors = 0;
    int print = 0;

    for (int i=0; i<n; i++) {

        if (isnan(a1[i]) || isnan(a2[i])) {
            errors++;
            if (print < 10) {
                print++;
                fprintf(stderr, "Error NaN detected at i=%d,\t a1= %10.7e \t a2= \t %10.7e\n",i,a1[i],a2[i]);
            }
        }

        float diff = (a1[i]-a2[i])/a1[i];
        if (diff > 1e-6f) {
            errors++;
            if (print < 10) {
                print++;
                fprintf(stderr, "Error detected at i=%d, \t a1= \t %10.7e \t a2= \t %10.7e \t rel_error=\t %10.7e\n",i,a1[i],a2[i],diff);
            }
        }
    }
    return errors;
}


int main() {
    int i; 
    int errors=0;
    float filter_sum = 0.0; 

    //allocate arrays and fill them
    float *input = (float *) malloc(input_height * input_width * sizeof(float));
    float *output1 = (float *) calloc(image_height * image_width, sizeof(float));
    float *output2 = (float *) calloc(image_height * image_width, sizeof(float));
    float *filter = (float *) malloc(filter_height * filter_width * sizeof(float));

    for (i=0; i< input_height * input_width; i++) {
        input[i] = (float) (i % SEED);
    }

    //This is specific for a W==H smoothening filteri, where W and H are odd.
    for (i=0; i<filter_height * filter_width; i++) { 
      filter[i] = 1.0;
    }

    for (i = filter_width+1 ; i < (filter_height-1) * filter_width; i++) {
        if (i % filter_width > 0 && i % filter_width < filter_width-1) {
            filter[i]+=1.0; 
        }
    }

    filter[filter_width*filter_height/2]=3.0;
    //end initialization

    // sum filter values
    for (i=0; i< filter_width*filter_height; i++) {
        filter_sum += filter[i]; 
    }


    //measure the GPU function
    convolutionCUDA(output2, input, filter, filter_sum);
    //measure the CPU function
    convolutionSeq(output1, input, filter, filter_sum);

    //check the result
    errors += compare_arrays(output1, output2, image_height*image_width);
    if (errors > 0) {
        printf("TEST FAILED! %d errors!\n", errors);
    } else {
        printf("TEST PASSED!\n");
    }


    free(filter);
    free(input);
    free(output1);
    free(output2);

    return 0;
}


