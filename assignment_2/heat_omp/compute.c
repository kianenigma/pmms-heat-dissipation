#include <sys/time.h>
#include "omp.h"
#include <math.h>
#include <stdlib.h>
#include "stdio.h"

#include "compute.h"
#define M_SQRT2    1.41421356237309504880

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

static void do_copy(size_t h, size_t w,
                    double (*restrict g)[h][w])
{
    size_t i;

    /* copy left and right column to opposite border */
    for (i = 0; i < h; ++i) {
        (*g)[i][w-1] = (*g)[i][1];
        (*g)[i][0] = (*g)[i][w-2];
    }
}

void do_compute(const struct parameters* p, struct results *r)
{
    size_t i, j;

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

    static const double c_cdir = 0.25 * M_SQRT2 / (M_SQRT2 + 1.0);
    static const double c_cdiag = 0.25 / (M_SQRT2 + 1.0);

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

    /* omp initial logs */
    printf("OMP :: max threads of systems %d | will use: %d\n", omp_get_max_threads(), p->nthreads);

    gettimeofday(&before, NULL);
    for (iter = 1; iter <= p->maxiter; ++iter)
    {
        double maxdiff = 0.0;

        /* swap source and destination */
        { void *tmp = src; src = dst; dst = tmp; }

        /* initialize halo on source */
        do_copy(h, w, src);

        /* compute */
#pragma omp parallel for \
        private(i, j)\
        schedule(static)\
        reduction(max: maxdiff)\
        num_threads(p->nthreads)
        for (i = 1; i < h - 1; ++i) {
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
        
        if ( maxdiff < p->threshold ) { break; }

        if ( p->printreports ) {
            double tmin = INFINITY, tmax = -INFINITY;
            double sum = 0.0;
            struct timeval after;

            /* We have said that the final reduction does not need to be included. */
            gettimeofday(&after, NULL);

            for (i = 1; i < h - 1; ++i) {
                for (j = 1; j < w - 1; ++j) {
                    double v = (*dst)[i][j];
                    double v_old = (*src)[i][j];

                    sum += v;
                    if (tmin > v) tmin = v;
                    if (tmax < v) tmax = v;
                }
            }

            r->niter = iter;
            r->maxdiff = maxdiff;
            r->tmin = tmin;
            r->tmax = tmax;
            r->tavg = sum / (p->N * p->M);

            r->time = (double)(after.tv_sec - before.tv_sec) +
                      (double)(after.tv_usec - before.tv_usec) / 1e6;

            report_results(p, r);

        }
    }

    /* report at end in all cases */
    fill_report(p, r, h, w, dst, src, iter, &before);

    free(c);
    free(g2);
    free(g1);

}
