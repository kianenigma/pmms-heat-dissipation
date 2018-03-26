#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include <sys/time.h>
#include "compute.h" 

static const double c_cdir = 0.25 * M_SQRT2 / (M_SQRT2 + 1.0);
static const double c_cdiag = 0.25 / (M_SQRT2 + 1.0);

static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        printf("cuda error [code %d][%s]\n", result, cudaGetErrorString(result));
        exit(1);
    }
}

static int index(int i, int j, int WIDTH) { return WIDTH*i + j; }

__device__ int _index(int row, int col, int WIDTH) { return (WIDTH)*(row+1) + (col+1); }

static void fill_report(size_t w, size_t h, double* dst, struct results* r, double global_maxdiff, int iter, struct timeval before, struct timeval after) {
    double tmin = INFINITY, tmax = -INFINITY;
    double sum = 0.0;
    int i, j;

    for (i = 1; i < h - 1 ; ++i) {
        for (j = 1; j < w - 1 ; ++j) {
            double v = dst[index(i,j,w)];
            sum += v;
            if (tmin > v) tmin = v;
            if (tmax < v) tmax = v;
        }
    }

    r->niter = iter;
    r->maxdiff = global_maxdiff;
    r->tmin = tmin;
    r->tmax = tmax;
    r->tavg = sum / ((w-2) * (h-2));
    r->time = (double)(after.tv_sec - before.tv_sec) + 
        (double)(after.tv_usec - before.tv_usec) / 1e6;
}

static void summary_matrix(size_t w, size_t h, const double *a) {
    int H;
    printf("################\n"); 
    H = 0;
    printf("%lf\t%lf\t%lf\t%lf  ....  %lf\t%lf\t%lf\t%lf\n",
        a[index(H,0,w)], a[index(H,1,w)], a[index(H,2,w)], a[index(H,3,w)],
        a[index(H, w-4,w)], a[index(H, w-3,w)], a[index(H, w-2,w)], a[index(H, w-1,w)]);
    H = 1;
    printf("%lf\t%lf\t%lf\t%lf  ....  %lf\t%lf\t%lf\t%lf\n",
        a[index(H,0,w)], a[index(H,1,w)], a[index(H,2,w)], a[index(H,3,w)],
        a[index(H, w-4,w)], a[index(H, w-3,w)], a[index(H, w-2,w)], a[index(H, w-1,w)]);
    H = 2;
    printf("%lf\t%lf\t%lf\t%lf  ....  %lf\t%lf\t%lf\t%lf\n",
        a[index(H,0,w)], a[index(H,1,w)], a[index(H,2,w)], a[index(H,3,w)],
        a[index(H, w-4,w)], a[index(H, w-3,w)], a[index(H, w-2,w)], a[index(H, w-1,w)]);
    H = 3;
    printf("%lf\t%lf\t%lf\t%lf  ....  %lf\t%lf\t%lf\t%lf\n",
        a[index(H,0,w)], a[index(H,1,w)], a[index(H,2,w)], a[index(H,3,w)],
        a[index(H, w-4,w)], a[index(H, w-3,w)], a[index(H, w-2,w)], a[index(H, w-1,w)]);

    printf("... \t ... \n");
    printf("... \t ... \n");
    
    H = h-4;
    printf("%lf\t%lf\t%lf\t%lf  ....  %lf\t%lf\t%lf\t%lf\n",
        a[index(H,0,w)], a[index(H,1,w)], a[index(H,2,w)], a[index(H,3,w)],
        a[index(H, w-4,w)], a[index(H, w-3,w)], a[index(H, w-2,w)], a[index(H, w-1,w)]);
    H = h-3;
    printf("%lf\t%lf\t%lf\t%lf  ....  %lf\t%lf\t%lf\t%lf\n",
        a[index(H,0,w)], a[index(H,1,w)], a[index(H,2,w)], a[index(H,3,w)],
        a[index(H, w-4,w)], a[index(H, w-3,w)], a[index(H, w-2,w)], a[index(H, w-1,w)]);
    H = h-2;
    printf("%lf\t%lf\t%lf\t%lf  ....  %lf\t%lf\t%lf\t%lf\n",
        a[index(H,0,w)], a[index(H,1,w)], a[index(H,2,w)], a[index(H,3,w)],
        a[index(H, w-4,w)], a[index(H, w-3,w)], a[index(H, w-2,w)], a[index(H, w-1,w)]);
    H = h-1;
    printf("%lf\t%lf\t%lf\t%lf  ....  %lf\t%lf\t%lf\t%lf\n",
        a[index(H,0,w)], a[index(H,1,w)], a[index(H,2,w)], a[index(H,3,w)],
        a[index(H, w-4,w)], a[index(H, w-3,w)], a[index(H, w-2,w)], a[index(H, w-1,w)]);
    printf("###################\n");
}
    
/**
 * Main cell update kernel. lauched with always 32 threads in both dimension for each block. The block
 * size is calculated based on input size. 
 *
*/
__global__ void cellUpdateKernel(double* src, double* dst, const double* cond, size_t w, size_t h, const size_t maxiter, double* maxdiff) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; 
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= h-2 || j >= w-2 ) { return; }

    /* Reduction of strength - pre computed indexes */
    const unsigned int prev_row_base = (i)*w; 
    const unsigned int row_base = prev_row_base+w;  
    const unsigned int next_row_base = row_base+w; 
    const unsigned int cell_base = row_base+j+1; 

    double weight = cond[cell_base];
    double restw = 1.0 - weight;
    double v, v_old;
    v_old = src[cell_base];

    v = weight * v_old +
    (
        src[next_row_base+j+1] + 
        src[prev_row_base+j+1] + 
        src[row_base+j+2] + 
        src[row_base+j]
        ) * (restw * c_cdir)
    +
    ( 
        src[prev_row_base+j] + 
        src[prev_row_base+j+2] +
        src[next_row_base+j] +
        src[next_row_base+j+2]
        ) * (restw * c_cdiag);

    dst[cell_base] = v;

    double diff = fabs(v - v_old);
    maxdiff[cell_base] = diff; 
}

/**
* Kernel used to mirror the first and last column
* Deplyed in the following size: 
*     dim3 mirror_dim_grid(1, GRID_DIM_Y, 1); 
*     dim3 mirror_dim_block(2, BLOCK_DIM, 1); 
*/
__global__ void mirrorKernel(double* dst, size_t w, size_t h) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; 
    unsigned int row_base = (i+1)*w; 
    if ( i >= h-2 ) { return; }

    /* swap firs and last column in parallel, if needed */
    if (threadIdx.x == 0) { 
        dst[row_base+0] = dst[row_base+w-2]; 
    }   
    if (threadIdx.x == 1) {
        dst[row_base+w-1] = dst[row_base+1]; 
    }
}


__global__ void diffUpdateKernel_sharedMem(size_t w, size_t h, double* maxdiff) {
    const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int block_resp_index = blockDim.x*threadIdx.y + threadIdx.x; 

    if ( i >= h-2 || j >= w-2 ) { return; }

    /* Load the entire matrix in parallel and sync each thread */
    extern __shared__ double shared_maxdiff[]; 
    shared_maxdiff[block_resp_index] = maxdiff[_index(i,j,w)];     
    __syncthreads(); 

    /* Reduce each row of the block horizantally */
    for (unsigned int s=(blockDim.x)/2; s>0; s>>=1) {
        if (threadIdx.x < s) {
            if (shared_maxdiff[block_resp_index+s] > shared_maxdiff[block_resp_index]) {
                shared_maxdiff[block_resp_index] = shared_maxdiff[block_resp_index+s]; 
            }
        }
        __syncthreads();
    }

    /* Reduce the first column vertically */
    if ( threadIdx.x == 0 ) {
        for (unsigned int s=(blockDim.y)/2; s>0; s>>=1) {
            if (threadIdx.y < s) {
                if (shared_maxdiff[blockDim.x*(threadIdx.y+s)] > shared_maxdiff[block_resp_index]) {
                    shared_maxdiff[block_resp_index] = shared_maxdiff[blockDim.x*(threadIdx.y+s)]; 
                }
            }
            __syncthreads();
        }
    }

    /* one thread writes the result back */
    if ( threadIdx.x == 0 && threadIdx.y == 0 ) {
        maxdiff[w+1+(blockIdx.x)+(gridDim.x*blockIdx.y)] = shared_maxdiff[0]; 
    }
}

__global__ void diffUpdateKernel_sharedMem_2(size_t w, size_t h, double* maxdiff) {
    // const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; 
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int block_resp_index = blockDim.x*threadIdx.y + threadIdx.x; 

    /* Load the entire matrix in parallel and sync each thread */
    extern __shared__ double shared_maxdiff[]; 
    shared_maxdiff[threadIdx.x] = maxdiff[_index(0,j,w)];     
    __syncthreads(); 

    /* Reduce one row of the block horizantally */
    for (unsigned int s=(blockDim.x)/2; s>0; s>>=1) {
        if (threadIdx.x < s) {
            if (shared_maxdiff[block_resp_index+s] > shared_maxdiff[block_resp_index]) {
                shared_maxdiff[block_resp_index] = shared_maxdiff[block_resp_index+s]; 
            }
        }
        __syncthreads();
    }

    /* one thread writes the result back */
    if ( threadIdx.x == 0 && threadIdx.y == 0 ) {
        maxdiff[w+1] = shared_maxdiff[0]; 
    }
}

__global__ void diffUpdateKernel(size_t w, size_t h, double* maxdiff) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; 
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= h-2 || j >= w-2 ) { return; }

    for (unsigned int s=(blockDim.x*gridDim.x)/2; s>0; s>>=1) {
        // if ( threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 )
        // printf("Thread 0 0 0 will compare %d to %d \n", j, j+s);
        if (j < s) {
            if (maxdiff[_index(i,j+s,w)] > maxdiff[_index(i,j,w)]) {
                maxdiff[_index(i,j,w)] = maxdiff[_index(i,j+s,w)]; 
            }
        }
        __syncthreads();
    }
}

__global__ void diffUpdateKernel_2(size_t w, size_t h, double* maxdiff) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; 
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i >= h-2 || j >= w-2 ) { return; }

    for (unsigned int s=(blockDim.y*gridDim.y)/2; s>0; s>>=1) {
        if (i < s) {
        //     if ( threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0 )
        // printf("2Thread 0 0 0 will compare %d to %d \n", i, i+s);
            if (maxdiff[_index(i+s,j,w)] > maxdiff[_index(i,j,w)]) {
                maxdiff[_index(i,j,w)] = maxdiff[_index(i+s,j,w)]; 
            }
        }
        __syncthreads();
    }
}

__global__ void GlobalcellUpdateKernel(double* src, double* dst, const double* cond, size_t w, size_t h, const size_t maxiter, int* block_flag) {
    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y; 
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    int it; 
    for (it = 1; it < maxiter; it++) {
        double weight = cond[_index(i,j,w)];
        double restw = 1.0 - weight;
        double v, v_old;
        v_old = src[_index(i,j,w)];

        v = weight * v_old +
        (
            src[_index(i+1,j,w)] + 
            src[_index(i-1,j,w)] + 
            src[_index(i,j+1,w)] + 
            src[_index(i,j-1,w)]
            ) * (restw * c_cdir)
        +
        ( 
            src[_index(i-1,j-1,w)] + 
            src[_index(i-1,j+1,w)] +
            src[_index(i+1,j-1,w)] +
            src[_index(i+1,j+1,w)]
            ) * (restw * c_cdiag);

        dst[_index(i,j,w)] = v; 

        /* swap firs and last column in parallel, if needed */
        if (blockIdx.x == 0 && threadIdx.x == 0) { 
            dst[_index(i, -1, w)] = dst[_index(i, w-3, w)]; 
        }   
        if ( threadIdx.x == (blockDim.x-1) && blockIdx.x == (gridDim.x-1) ) {
            dst[_index(i, w-2, w)] = dst[_index(i, 0, w)]; 
        }

        /* Ensure that all threads in the current block are done with the update */
        __syncthreads(); 

        /* Ensure all perations in this block are written to memory */
        __threadfence_block(); 

        /* mark the iteration flag if this block */
        if ( threadIdx.x == 0 && threadIdx.y == 0 ) {
            // printf("Master thread of block %d %d. DONE\n", blockIdx.y, blockIdx.x);
            // TODO: this sould not be atomic, right? 
            block_flag[blockIdx.x + blockIdx.y * gridDim.x] = it; 
        }

        /* wait until all blocks are merked */
        while(1) {
            for (int p = 0; p < gridDim.x*gridDim.y; p++) {
                if (block_flag[p] != it ) { 
                    continue; 
                }
            } 
            break; 
        }

        printf("Thread %d done with iteration\n", it);
        /* all threads swap the pointers for the next iteration */
        { double *tmp = src; src = dst; dst = tmp; }
        // break;

    }
}


extern "C" void cuda_do_compute(const struct parameters* p, struct results *r) {
    struct timeval before, after;

    const size_t N = p->N; 
    const size_t M =  p->M; 

    printf("ORIGINAL DIMENSTIONS [%zd %zd]\n", N, M); 

    const size_t w = M+2; 
    const size_t h = N+2; 

    const size_t MALLOC_VAL = w*h; 

    /* It is important for thread batches to be multiplies of 32 */
    unsigned const int BLOCK_DIM = 32; 
    unsigned int GRID_DIM_X; 
    unsigned int GRID_DIM_Y;

    /* Find the minimum number of blocks that is bigger than the data size */
    GRID_DIM_X = ceil((float)M/BLOCK_DIM); 
    GRID_DIM_Y = ceil((float)N/BLOCK_DIM);  

    const double *tinit = p->tinit; 
    const double *cond = p->conductivity;

    double *h_dst   = (double*)malloc(MALLOC_VAL*sizeof(double)); 
    double *h_src   = (double*)malloc(MALLOC_VAL*sizeof(double)); 
    double *h_cond  = (double*)malloc(MALLOC_VAL*sizeof(double)); 

    if (!h_src || !h_dst || !h_cond) {
        printf("malloc failed\n");
        exit(1);
    }

    double *d_src;
    double *d_dst;  
    double *d_cond; 

    /* Initialize value, mirrors, halo grids and stuff */
    int i, j;
    for (i = 1; i < h - 1; ++i) {
        for (j = 1; j < w - 1; ++j)
        {
            h_src[index(i,j,w)]  = tinit[index(i-1,j-1,M)];
            h_dst[index(i,j,w)]  = tinit[index(i-1,j-1,M)];
            h_cond[index(i,j,w)] = cond [index(i-1,j-1,M)];
        }
    }

    /* smear outermost row to border */
    for (j = 1; j < w-1; ++j) {
        h_src[0 + j] = h_src[0 + j] = h_src[1*w + j];
        h_dst[0 + j] = h_dst[0 + j] = h_dst[1*w + j];
        h_src[(h-1)*w + j] = h_src[(h-1)*w + j] = h_src[(h-2)*w + j];
        h_dst[(h-1)*w + j] = h_dst[(h-1)*w + j] = h_dst[(h-2)*w + j];
    }

    // mirror 
    for (i = 0; i < h ; ++i) {
        // column w-1 
        h_src[i*w + w-1] = h_src[i*w + 1];
        h_dst[i*w + w-1] = h_dst[i*w + 1];
        // column 0 
        h_src[i*w + 0]   = h_src[i*w + w-2];
        h_dst[i*w + 0]   = h_dst[i*w + w-2];
    }
    
    printf("GRID_DIM= [%d %d] | BLOCK_DIM=[%d %d] | SAHRED_MEM_SIZE %ld\n", GRID_DIM_X, GRID_DIM_Y, BLOCK_DIM, BLOCK_DIM, BLOCK_DIM*BLOCK_DIM*sizeof(double));

    dim3 update_dim_grid(GRID_DIM_X, GRID_DIM_Y, 1); 
    dim3 update_dim_block(BLOCK_DIM, BLOCK_DIM, 1); 

    dim3 mirror_dim_grid(1, GRID_DIM_Y, 1); 
    dim3 mirror_dim_block(2, BLOCK_DIM, 1); 

    dim3 maxdiff_dim_grid(GRID_DIM_X, GRID_DIM_Y, 1); 
    dim3 maxdiff_dim_block(BLOCK_DIM, BLOCK_DIM, 1); 

    dim3 maxdiff_2_dim_grid(1, GRID_DIM_Y, 1); 
    dim3 maxdiff_2_dim_block(1, BLOCK_DIM, 1);

    dim3 maxdiff_2_shared_dim_grid(1, 1, 1); 
    dim3 maxdiff_2_shared_dim_block(GRID_DIM_X*GRID_DIM_Y, 1, 1); 

    /* start time */
    gettimeofday(&before, NULL); 

    /* Init space for src, dst, cond in GPU memeory */
    checkCudaCall(cudaMalloc((void **) &d_src,  MALLOC_VAL*sizeof(double))); 
    checkCudaCall(cudaMalloc((void **) &d_dst,  MALLOC_VAL*sizeof(double))); 
    checkCudaCall(cudaMalloc((void **) &d_cond, MALLOC_VAL* sizeof(double))); 

    /* Copy everything to device memory */
    checkCudaCall(cudaMemcpy(d_src, h_src,   MALLOC_VAL*sizeof(double), cudaMemcpyHostToDevice)); 
    checkCudaCall(cudaMemcpy(d_dst, h_dst,   MALLOC_VAL*sizeof(double), cudaMemcpyHostToDevice)); 
    checkCudaCall(cudaMemcpy(d_cond, h_cond, MALLOC_VAL*sizeof(double), cudaMemcpyHostToDevice)); 

    /* maxdiff variables */
    double *h_maxdiff = (double *)malloc(MALLOC_VAL*sizeof(double)); 
    for (int i = 0; i < h*w; i++) { h_maxdiff[i] = 0; }
    double *d_maxdiff; 
    checkCudaCall(cudaMalloc((void **) &d_maxdiff, MALLOC_VAL*sizeof(double))); 
    checkCudaCall(cudaMemcpy(d_maxdiff, h_maxdiff, MALLOC_VAL*sizeof(double), cudaMemcpyHostToDevice)); 

    double *global_maxdiff = (double*) malloc(sizeof(double)); 
    int it; 
    for (it = 0; it < p->maxiter; it++) {
        /* All cells will be updated in d_dest */
        cellUpdateKernel<<<update_dim_grid, update_dim_block>>>(
            d_src, d_dst, d_cond,
            w, h, p->maxiter, d_maxdiff); 

        /* update first and last column */
        // TODO: should be faster with two kernels with no IF inside? 
        mirrorKernel<<<mirror_dim_grid, mirror_dim_block>>>(d_dst, w, h); 

        /* calculate diff,  */
        // TODO: maybe would be more optimzied with an initial kernel half of the size in each row (see slides) 

        // diffUpdateKernel
        //     <<<maxdiff_dim_grid, maxdiff_dim_block>>>
        //     (w, h, d_maxdiff);  
        // diffUpdateKernel_2
        //     <<<maxdiff_2_dim_grid, maxdiff_2_dim_block>>>
        //     (w, h, d_maxdiff);

        diffUpdateKernel_sharedMem
            <<<maxdiff_dim_grid, maxdiff_dim_block, BLOCK_DIM*BLOCK_DIM*sizeof(double)>>>
            (w, h, d_maxdiff);

        diffUpdateKernel_sharedMem_2
            <<<maxdiff_2_shared_dim_grid, maxdiff_2_shared_dim_block, GRID_DIM_X*GRID_DIM_Y*sizeof(double)>>>
            (w, h, d_maxdiff);

        // DEBUG 
        // printf("result at end of iter %d\n", it);
        // checkCudaCall(cudaMemcpy(h_dst, d_dst, MALLOC_VAL*sizeof(double), cudaMemcpyDeviceToHost)); 
        // summary_matrix(w, h, h_dst);
        // checkCudaCall(cudaMemcpy(h_maxdiff, d_maxdiff, w*h*sizeof(double), cudaMemcpyDeviceToHost)); 
        // printf("maxdiff at end of iter %d\n", it);
        // summary_matrix(w, h, h_maxdiff);

        /* Copy just one value from maxdiff kernel out */
        checkCudaCall(cudaMemcpy(global_maxdiff, d_maxdiff+w+1, sizeof(double), cudaMemcpyDeviceToHost)); 
        if ( *global_maxdiff < p->threshold ) { break; }

        if ( p->printreports ) { 
            if ( it % p->period == 0 ) {
                checkCudaCall(cudaMemcpy(h_dst, d_dst, MALLOC_VAL*sizeof(double), cudaMemcpyDeviceToHost));
                fill_report(w, h, h_dst, r, *global_maxdiff, it, before, after);  
                // report_results(p, r);
            }
        }
        /* swap pointers for next iteration, if exists */
        // TODO: this will cause miskates for the last iteration in case of maxiter termination
        { double *tmp = d_src; d_src = d_dst; d_dst = tmp; }
    }

    /*  Sync and fetch the latest results */
    cudaDeviceSynchronize();   
    checkCudaCall(cudaGetLastError()); 
    checkCudaCall(cudaMemcpy(h_dst, d_dst, MALLOC_VAL*sizeof(double), cudaMemcpyDeviceToHost));
    gettimeofday(&after, NULL);  
    // summary_matrix(w, h, h_dst);
    fill_report(w, h, h_dst, r, *global_maxdiff, it, before, after);     

    /* cleanup device */
    checkCudaCall(cudaFree(d_dst)); 
    checkCudaCall(cudaFree(d_cond)); 
    checkCudaCall(cudaFree(d_src)); 
    checkCudaCall(cudaFree(d_maxdiff)); 

    /* cleanup host */
    free(h_src);
    free(h_dst); 
    free(h_cond); 
    free(h_maxdiff);
}
