#include <sys/time.h>
#include <math.h>
#include <stdlib.h>
#include "stdio.h"
#include <pthread.h>
#include "compute.h"
#include "pthread_barrier.h"

#define M_SQRT2 1.41421356237309504880

// TODO: this is doing one iteration less than expected.
// TODO: parameter passing of restrict pointer uses casting. This might disable optimization. check it.
// TODO: see int_fast32_t data type

static const double c_cdir = 0.25 * M_SQRT2 / (M_SQRT2 + 1.0);
static const double c_cdiag = 0.25 / (M_SQRT2 + 1.0);

pthread_barrier_t barrier;

typedef struct thread_params{
    int start_idx;
    int end_idx;
    int copy_start_idx;
    int copy_end_idx;
    int id;
    int *num_threads;
    double *diff_buffer;
    size_t *iter;
    double ***src_ptr;
    double ***dst_ptr;
    double ***c_ptr;
    struct results* results_ptr;
    struct parameters* parameters_ptr;
    struct timeval *before;


} thread_params;


/**
 * Utility function to print a summary of the matrix
 */
static void print_matrix(size_t h, size_t w, double (*restrict a)[h][w]) {
    int i, j;
    printf("######################\n");
    i = 0;
    printf("%lf\t%lf\t%lf ...  %lf\t%lf\t%lf\n", (*a)[i][0], (*a)[i][1], (*a)[i][2], (*a)[i][w-3], (*a)[i][w-2], (*a)[i][w-1]);
    i = 1;
    printf("%lf\t%lf\t%lf ...  %lf\t%lf\t%lf\n", (*a)[i][0], (*a)[i][1], (*a)[i][2], (*a)[i][w-3], (*a)[i][w-2], (*a)[i][w-1]);
    i = 2;
    printf("%lf\t%lf\t%lf ...  %lf\t%lf\t%lf\n", (*a)[i][0], (*a)[i][1], (*a)[i][2], (*a)[i][w-3], (*a)[i][w-2], (*a)[i][w-1]);
    i = 3;
    printf("%lf\t%lf\t%lf ...  %lf\t%lf\t%lf\n", (*a)[i][0], (*a)[i][1], (*a)[i][2], (*a)[i][w-3], (*a)[i][w-2], (*a)[i][w-1]);
    i = 4;
    printf("%lf\t%lf\t%lf ...  %lf\t%lf\t%lf\n", (*a)[i][0], (*a)[i][1], (*a)[i][2], (*a)[i][w-3], (*a)[i][w-2], (*a)[i][w-1]);
    printf("....\n");
    printf("....\n");
    printf("....\n");
    printf("....\n");
    i = h-5;
    printf("%lf\t%lf\t%lf ...  %lf\t%lf\t%lf\n", (*a)[i][0], (*a)[i][1], (*a)[i][2], (*a)[i][w-3], (*a)[i][w-2], (*a)[i][w-1]);
    i = h-4;
    printf("%lf\t%lf\t%lf ...  %lf\t%lf\t%lf\n", (*a)[i][0], (*a)[i][1], (*a)[i][2], (*a)[i][w-3], (*a)[i][w-2], (*a)[i][w-1]);
    i = h-3;
    printf("%lf\t%lf\t%lf ...  %lf\t%lf\t%lf\n", (*a)[i][0], (*a)[i][1], (*a)[i][2], (*a)[i][w-3], (*a)[i][w-2], (*a)[i][w-1]);
    i = h-2;
    printf("%lf\t%lf\t%lf ...  %lf\t%lf\t%lf\n", (*a)[i][0], (*a)[i][1], (*a)[i][2], (*a)[i][w-3], (*a)[i][w-2], (*a)[i][w-1]);
    i = h-1;
    printf("%lf\t%lf\t%lf ...  %lf\t%lf\t%lf\n", (*a)[i][0], (*a)[i][1], (*a)[i][2], (*a)[i][w-3], (*a)[i][w-2], (*a)[i][w-1]);
    printf("######################\n");
}

/**
 * Utility function to print all attributes of the thread
 * @param attr the attribute object to be displayed
 */
static void display_pthread_attr(pthread_attr_t *attr, char *prefix) {
    int s, i;
    size_t v;
    void *stkaddr;
    struct sched_param sp;

    s = pthread_attr_getdetachstate(attr, &i);
    if (s != 0) { printf("error while printing thread attribute\n"); }
    printf("%sDetach state        = %s\n", prefix,
           (i == PTHREAD_CREATE_DETACHED) ? "PTHREAD_CREATE_DETACHED" :
           (i == PTHREAD_CREATE_JOINABLE) ? "PTHREAD_CREATE_JOINABLE" :
           "???");

    s = pthread_attr_getscope(attr, &i);
    if (s != 0) { printf("error while printing thread attribute\n"); }
    printf("%sScope               = %s\n", prefix,
           (i == PTHREAD_SCOPE_SYSTEM)  ? "PTHREAD_SCOPE_SYSTEM" :
           (i == PTHREAD_SCOPE_PROCESS) ? "PTHREAD_SCOPE_PROCESS" :
           "???");

    s = pthread_attr_getinheritsched(attr, &i);
    if (s != 0) { printf("error while printing thread attribute\n"); }
    printf("%sInherit scheduler   = %s\n", prefix,
           (i == PTHREAD_INHERIT_SCHED)  ? "PTHREAD_INHERIT_SCHED" :
           (i == PTHREAD_EXPLICIT_SCHED) ? "PTHREAD_EXPLICIT_SCHED" :
           "???");

    s = pthread_attr_getschedpolicy(attr, &i);
    if (s != 0) { printf("error while printing thread attribute\n"); }
    printf("%sScheduling policy   = %s\n", prefix,
           (i == SCHED_OTHER) ? "SCHED_OTHER" :
           (i == SCHED_FIFO)  ? "SCHED_FIFO" :
           (i == SCHED_RR)    ? "SCHED_RR" :
           "???");

    s = pthread_attr_getschedparam(attr, &sp);
    if (s != 0) { printf("error while printing thread attribute\n"); }
    printf("%sScheduling priority = %d\n", prefix, sp.sched_priority);

    s = pthread_attr_getguardsize(attr, &v);
    if (s != 0) { printf("error while printing thread attribute\n"); }
    printf("%sGuard size          = %zd bytes\n", prefix, v);

    s = pthread_attr_getstack(attr, &stkaddr, &v);
    if (s != 0) { printf("error while printing thread attribute\n"); }
    printf("%sStack address       = %p\n", prefix, stkaddr);
    printf("%sStack size          = 0x%zx bytes\n", prefix, v);
}

/**
 * Print the results
 */
static inline int fill_report(const struct parameters *p, struct results *r, size_t h, size_t w,
                              double (*restrict a)[h][w], double (*restrict b)[h][w], size_t iter,
                              struct timeval *before, struct timeval *after) {
    /* compute min/max/avg */
    double tmin = INFINITY, tmax = -INFINITY;
    double sum = 0.0;
    double maxdiff = 0.0;

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

    r->time = (double)(after->tv_sec - before->tv_sec) +
              (double)(after->tv_usec - before->tv_usec) / 1e6;

    return (maxdiff >= p->threshold) ? 0 : 1;
}

/**
 * Common function for all threads
 * @param p a pointer to thread_params struct
 * @return
 */
void *thread_proc(void *p) {
    thread_params *params = (thread_params*)p;
    size_t iter = 0;
    int i, j;
    double threshold = params->parameters_ptr->threshold;
    size_t maxiter = params->parameters_ptr->maxiter;
    size_t w = params->parameters_ptr->M + 2;
    size_t h = params->parameters_ptr->N + 2;
    size_t printreport = params->parameters_ptr->printreports, period = params->parameters_ptr->period;
    int start_idx = params->start_idx;
    int end_idx = params->end_idx;
    struct timeval *before = params->before;

    double (*restrict src)[h][w] = params->src_ptr;
    double (*restrict dst)[h][w] = params->dst_ptr;
    double (*restrict c)[h][w] = params->c_ptr;

    for (iter = 1; iter <= maxiter; ++iter) {

        /* note that this is pointer swap (local pointer that have shared values), each thread must do this */
        { void *tmp = src; src = dst; dst = tmp; }

        /* Copy last and first column */
        for (i = params->copy_start_idx; i < params->copy_end_idx ; i++) {
            (*src)[i][w-1] = (*src)[i][1];
            (*src)[i][0] = (*src)[i][w-2];
        }

        pthread_barrier_wait(&barrier);

        double maxdiff = 0.0;

        /* compute */
        for (i = start_idx; i < end_idx  ; ++i) {
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

        /* wait for all threads to write local max diff */
        pthread_barrier_wait(&barrier);

        /* compute global max diff */
        double global_diff = 0.0;
        for (i = 0; i < *params->num_threads; i++) {
            if (params->diff_buffer[i] > global_diff) {
                global_diff = params->diff_buffer[i];
            }
        }

        if (global_diff < threshold) {
            // break
            if (params->id == 0) {*params->iter = iter; }
            break;
        }


        /* thread 0 will print if needed */
        if ( printreport ) {
            if (params->id == 0) {
                if ( iter % period == 0 ) {
                    double tmin = INFINITY, tmax = -INFINITY;
                    double sum = 0.0;
                    struct timeval after;

                    /* We have said that the final reduction does not need to be included. */
                    gettimeofday(&after, NULL);
                    for (i = 1; i < h - 1 ; ++i) {
                        for (j = 1; j < w - 1 ; ++j) {
                            double v = (*dst)[i][j];
                            sum += v;
                            if (tmin > v) tmin = v;
                            if (tmax < v) tmax = v;
                        }
                    }

                    params->results_ptr->niter = iter;
                    params->results_ptr->maxdiff = global_diff;
                    params->results_ptr->tmin = tmin;
                    params->results_ptr->tmax = tmax;
                    params->results_ptr->tavg = sum / ((w-2) * (h-2));

                    params->results_ptr->time = (double)(after.tv_sec - before->tv_sec) +
                                                (double)(after.tv_usec - before->tv_usec) / 1e6;

                    report_results(params->parameters_ptr, params->results_ptr);
                }
            }
        }
    }
}

void do_compute(const struct parameters* p, struct results *r)
{
    printf("Running Pthread Parallel %zd threads\n", p->nthreads);
    int NUM_THREADS = (int)p->nthreads;
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

    struct timeval before, after;

    /* set initial temperatures and conductivities */
    for (i = 1; i < h - 1; ++i)
        for (j = 1; j < w - 1; ++j)
        {
            (*g1)[i][j] = (*tinit)[i-1][j-1];
            (*g2)[i][j] = (*tinit)[i-1][j-1];
            (*c)[i][j] = (*cinit)[i-1][j-1];
        }

    for (i = 0; i < h ; ++i) {
        (*g1)[i][w-1] = (*g1)[i][1];
        (*g1)[i][0] = (*g1)[i][w-2];
    }

    /* smear outermost row to border */
    for (j = 1; j < w-1; ++j) {
        (*g1)[0][j] = (*g2)[0][j] = (*g1)[1][j];
        (*g1)[h-1][j] = (*g2)[h-1][j] = (*g1)[h-2][j];
    }


    /* compute */
    size_t iter = 1;
    double (*restrict src)[h][w] = g2;
    double (*restrict dst)[h][w] = g1;


    /* Thread id array */
    int thread_ids[NUM_THREADS];
    pthread_t _thread_ids[NUM_THREADS];

    /* Thread argument variables  */
    int threads_start_idx[NUM_THREADS];
    int threads_end_idx[NUM_THREADS];
    int thread_cp_start_idx[NUM_THREADS];
    int thread_cp_end_idx[NUM_THREADS];

    double *diff_buffer = malloc(NUM_THREADS* sizeof(double));
    size_t _iter = 0;

    thread_params params_array[NUM_THREADS];

    // Split computation/copy space among threads
    int thread_row_counts[NUM_THREADS];
    int row_per_thread = p->M / NUM_THREADS;
    int remainder = p->M - (NUM_THREADS*row_per_thread);
    for (int t = 0; t < NUM_THREADS; t++) { thread_row_counts[t] = row_per_thread; }
    int turn = 0;
    while (remainder != 0 ) {
        thread_row_counts[turn%NUM_THREADS]++;
        remainder--;
        turn++;
    }
    int rows_assigned = 0;
    for (int t = 0; t < NUM_THREADS; t++) {
        threads_start_idx[t] = 1 + rows_assigned ;
        threads_end_idx[t] = 1 + rows_assigned + thread_row_counts[t];

        thread_cp_end_idx[t] = threads_end_idx[t];
        thread_cp_start_idx[t] = threads_start_idx[t];

        rows_assigned += thread_row_counts[t];
    }

    thread_cp_start_idx[0] = 0;
    thread_cp_end_idx[NUM_THREADS-1] = h;


    for (i = 0; i < NUM_THREADS; i++) {
        printf("[Thread %d] :: %d -> %d [copy: %d -> %d][weigth=%d]\n", i, threads_start_idx[i], threads_end_idx[i], thread_cp_start_idx[i], thread_cp_end_idx[i], threads_end_idx[i]-threads_start_idx[i]);
        // assign id of the thread.
        thread_ids[i] = i;
    }


    /* Init barrier and attribute */
    pthread_barrier_init(&barrier, NULL, NUM_THREADS);
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);

    printf("\n# Thread attributes:\n");
    display_pthread_attr(&attr, "");

    /* Initialize 4 threads with their parameters*/
    for (i = 0; i < NUM_THREADS; i++) {
        thread_params* params = &params_array[i];
        params->id = thread_ids[i];
        params->start_idx = threads_start_idx[i];
        params->end_idx = threads_end_idx[i];
        params->copy_start_idx = thread_cp_start_idx[i];
        params->copy_end_idx = thread_cp_end_idx[i];
        params->src_ptr = (double***)src;
        params->dst_ptr = (double***)dst;
        params->c_ptr = (double***)c;
        params->diff_buffer = diff_buffer;
        params->iter = &_iter;
        params->num_threads = &NUM_THREADS;
        params->results_ptr = r;
        params->parameters_ptr = p;

        params->before = &before;


        pthread_create(&(_thread_ids[i]), &attr, (void*) thread_proc, params);
        if (i == 0) {
            gettimeofday(&before, NULL);
        }
    }


    /* Wait for all of them to finish */
    for (i = 0; i < NUM_THREADS; i++) {
        pthread_join(_thread_ids[i], NULL);
        if (i == 0) {
            iter = *params_array[i].iter;
        }
    }

    gettimeofday(&after, NULL);

    if (iter == 0) { iter = p->maxiter; }

    /* Do one last swap to get the updates of the last iteration */
    { void *tmp = src; src = dst; dst = tmp; }

    /* report at end in all cases */
    fill_report(p, r, h, w, dst, src, iter, &before, &after);

    free(c);
    free(g2);
    free(g1);
    free(diff_buffer);
}
