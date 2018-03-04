#include <sys/time.h>
#include <math.h>
#include <stdlib.h>
#include "stdio.h"
#include <pthread.h>
#include "compute.h"
#include "pthread_barrier.h"


#define NUM_THREADS 4

static const double c_cdir = 0.25 * M_SQRT2 / (M_SQRT2 + 1.0);
static const double c_cdiag = 0.25 / (M_SQRT2 + 1.0);
pthread_barrier_t barrier;

typedef struct thread_params{
    int start_idx;
    int end_idx;
    int id;
    double threshold;
    size_t maxiter;
    double *src_ptr;
    double *dst_ptr;
    double *c_ptr;
    double *diff_buffer;
    int *diff_flag;
    size_t h, w;

} thread_params;
/* Does the reduction step and return if the convergence has setteled */
static inline int fill_report(const struct parameters *p, struct results *r,
                              size_t h, size_t w,
                              double (*restrict a)[h][w],
                              double (*restrict b)[h][w],
                              double iter,
                              struct timeval *before)
{
    /* compute min/max/avg */
    double tmin = INFINITY, tmax = -INFINITY;
    double sum = 0.0;
    double maxdiff = 0.0;
    struct timeval after;

    /* We have said that the final reduction does not need to be included. */
    gettimeofday(&after, NULL);

    for (size_t i = 1; i < h - 1; ++i)
        for (size_t j = 1; j < w - 1; ++j)
        {
            double v = (*a)[i][j];
            double v_old = (*b)[i][j];
            double diff = fabs(v - v_old);
            sum += v;
            if (tmin > v) tmin = v;
            if (tmax < v) tmax = v;
            if (diff > maxdiff) maxdiff = diff;
        }

    r->niter = iter;
    r->maxdiff = maxdiff;
    r->tmin = tmin;
    r->tmax = tmax;
    r->tavg = sum / (p->N * p->M);

    r->time = (double)(after.tv_sec - before->tv_sec) +
              (double)(after.tv_usec - before->tv_usec) / 1e6;

    return (maxdiff >= p->threshold) ? 0 : 1;
}


void *thread_proc(void *p) {
    thread_params *params = (thread_params*)p;
    int iter = 0;
    int i, j;
    double threshold = params->threshold;
    size_t maxiter = params->maxiter;
    size_t w = params->w;
    size_t h = params->h;
    int start_idx = params->start_idx;
    int end_idx = params->end_idx;

    // TODO: fix the type mismatch here.
    double (*restrict src)[h][w] = params->src_ptr;
    double (*restrict dst)[h][w] = params->dst_ptr;
    double (*restrict c)[h][w] = params->c_ptr;

    for (iter = 1; iter <= maxiter; ++iter)
    {
        double maxdiff = 0.0;

        /* swap source and destination */
        { void *tmp = src; src = dst; dst = tmp; }

        /* initialize halo on source */
        for (i = start_idx; i < end_idx - 1; ++i) {
            (*src)[i][w-1] = (*src)[i][1];
            (*src)[i][0] = (*src)[i][w-2];
        }

        /* compute */
        for (i = start_idx; i < end_idx - 1; ++i) {

            for (j = 1; j < w - 1; ++j)
            {
                double w = (*c)[i][j];
                double restw = 1.0 - w;
                double v, v_old;
                v_old = (*src)[i][j];

                v = w * v_old +
                    ((*src)[i+1][j  ] + (*src)[i-1][j  ] +
                     (*src)[i  ][j+1] + (*src)[i  ][j-1]) * (restw * c_cdir) +
                    ((*src)[i-1][j-1] + (*src)[i-1][j+1] +
                     (*src)[i+1][j-1] + (*src)[i+1][j+1]) * (restw * c_cdiag);

                double diff = fabs(v - v_old);
                if (diff > maxdiff) maxdiff = diff;
                (*dst)[i][j] = v;
            }
        }

        /* write local max diff */
        params->diff_buffer[params->id] = maxdiff;
        pthread_barrier_wait(&barrier);

        /* first thread will reduce all diffs and set the flag */
        if ( params->id == 0 ) {
            double global_diff = 0.0;
            for (i = 0; i < NUM_THREADS; i++) {
                if (params->diff_buffer[i] > global_diff) {
                    global_diff = params->diff_buffer[i];
                }
            }
            if (global_diff < threshold) {
                // break
                *params->diff_flag = 1;
            }
            else {
                // continue
                *params->diff_flag = 0;
            }
        }

        pthread_barrier_wait(&barrier);
        if ( *params->diff_flag == 1 ) {
            break;
        }

//        if ( p->printreports ) {
//            iter--;
//            double tmin = INFINITY, tmax = -INFINITY;
//            double sum = 0.0;
//            struct timeval after;
//
//            /* We have said that the final reduction does not need to be included. */
//            gettimeofday(&after, NULL);
//
//            for (i = 1; i < h - 1; ++i) {
//                for (j = 1; j < w - 1; ++j) {
//                    double v = (*dst)[i][j];
//                    double v_old = (*src)[i][j];
//
//                    sum += v;
//                    if (tmin > v) tmin = v;
//                    if (tmax < v) tmax = v;
//                }
//            }
//
//            r->niter = iter;
//            r->maxdiff = maxdiff;
//            r->tmin = tmin;
//            r->tmax = tmax;
//            r->tavg = sum / (p->N * p->M);
//
//            r->time = (double)(after.tv_sec - before.tv_sec) +
//                      (double)(after.tv_usec - before.tv_usec) / 1e6;
//
//            report_results(p, r);
//
//        }
    }

    if (params->id == 0) {
        *params->diff_flag = iter;
    }

}

void do_compute(const struct parameters* p, struct results *r)
{
    printf("Running Pthread Parallel\n");
    int i, j;

    /* alias input parameters */
    const double (*restrict tinit)[p->N][p->M] = (const double (*)[p->N][p->M])p->tinit;
    const double (*restrict cinit)[p->N][p->M] = (const double (*)[p->N][p->M])p->conductivity;

    /* allocate grid data */
    const size_t h = p->N + 2;
    const size_t w = p->M + 2;
    double (*restrict g1)[h][w] = malloc(h * w * sizeof(double));
    double (*restrict g2)[h][w] = malloc(h * w * sizeof(double));

    /* allocate halo for conductivities */
    double (*restrict c)[h][w] = malloc(h * w * sizeof(double));

    struct timeval before;

    /* set initial temperatures and conductivities */
    for (i = 1; i < h - 1; ++i)
        for (j = 1; j < w - 1; ++j)
        {
            (*g1)[i][j] = (*tinit)[i-1][j-1];
            (*c)[i][j] = (*cinit)[i-1][j-1];
        }

    /* smear outermost row to border */
    for (j = 1; j < w-1; ++j) {
        (*g1)[0][j] = (*g2)[0][j] = (*g1)[1][j];
        (*g1)[h-1][j] = (*g2)[h-1][j] = (*g1)[h-2][j];
    }

    /* compute */
    size_t iter;
    double (*restrict src)[h][w] = g2;
    double (*restrict dst)[h][w] = g1;

    /* Thread id array */
    int thread_ids[NUM_THREADS];
    pthread_t _thread_ids[NUM_THREADS];

    int rows_per_thread = (int)h / NUM_THREADS;
    int threads_start_idx[NUM_THREADS];
    int threads_end_idx[NUM_THREADS];
    double *diff_buffer = malloc(NUM_THREADS* sizeof(double));
    int *diff_flag = malloc(sizeof(int)); *diff_flag = 0;
    thread_params params_array[NUM_THREADS];

    printf("Average row per thread %d. Thread Index max:\n", rows_per_thread);
    for (i = 0; i < NUM_THREADS; i++) {
        // TODO: this part should do as much as possible to make the rows for each thread balanced.
        if (i < NUM_THREADS-1) {
            threads_start_idx[i] = rows_per_thread*i + 1;
            threads_end_idx[i] = threads_start_idx[i] + rows_per_thread + 1;
        }
        else {
            threads_start_idx[i] = rows_per_thread*i + 1;
            threads_end_idx[i] = (int)h;
        }

        printf("[Thread %d] :: %d -> %d [weigth=%d]\n", i, threads_start_idx[i], threads_end_idx[i], threads_end_idx[i]-threads_start_idx[i]);

        // assign id of the thread.
        thread_ids[i] = i;
    }

    /* Init barrier */
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);

    /* Initialize 4 threads with their parameters*/
    for (i = 0; i < NUM_THREADS; i++) {
        // TODO: this must go inside an array so that we can free it an the end. Not a big problem for now
        thread_params* params = &params_array[i];
        params->id = thread_ids[i];
        params->start_idx = threads_start_idx[i];
        params->end_idx = threads_end_idx[i];
        params->maxiter = p->maxiter;
        params->threshold = p->threshold;
        params->src_ptr = src;
        params->dst_ptr = dst;
        params->c_ptr = c;
        params->h = h;
        params->w = w;
        params->diff_buffer = diff_buffer;
        params->diff_flag = diff_flag;


        pthread_create(&(_thread_ids[i]), NULL, (void*) thread_proc, params);

    }

    gettimeofday(&before, NULL);

    /* Wait for all of them to finish */
    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join(_thread_ids[i], NULL);
        if (i == 0) {
            iter = (size_t)*params_array[i].diff_flag;
        }
        printf("Thread %d done\n", thread_ids[i]);
    }
    
    /* report at end in all cases */
    iter--;
    fill_report(p, r, h, w, dst, src, iter, &before);

    free(c);
    free(g2);
    free(g1);

}
