#include <sys/time.h>
#include "omp.h"
#include <math.h>
#include <stdlib.h>
#include "stdio.h"
#include "time.h"

#include "compute.h"
#define M_SQRT2    1.41421356237309504880

long int timeval_subtract(struct timeval *t2, struct timeval *t1)
{
     return (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);

}

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

void do_compute_seq(const struct parameters* p, struct results *r)
{
    printf("Running Sequential\n");
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

    struct timeval t1, t2;
    long int par, seq;
    par = 0;
    seq = 0;

    gettimeofday(&before, NULL);
    for (iter = 1; iter <= p->maxiter; ++iter)
    {
        gettimeofday(&t1, NULL);
        double maxdiff = 0.0;

        /* swap source and destination */
        { void *tmp = src; src = dst; dst = tmp; }

        /* initialize halo on source */
        do_copy(h, w, src);

        gettimeofday(&t2, NULL);
        seq = seq + timeval_subtract(&t2, &t1);

        gettimeofday(&t1, NULL);
        /* compute */
        for (i = 1; i < h - 1; ++i) {
            for (j = 1; j < w - 1; ++j)
            {
                double w = (*c)[i][j];
                double restw = 1.0 - w;
                double v, v_old;
                v_old = (*src)[i][j];

                /*
                 * Flops:
                 * restw computation = 1
                 * 3 addition of neighboars times 2 = 6
                 * 2 multiply of restw to dir/diag weight = 2
                 * the sum of last two lines = 1
                 * multiply of w and v_old = 1
                 * fabs = 1
                 * compare = 1
                 * main sum = 1
                 *
                 * ### Total = 11
                 *
                 * Mem:
                 * in update loop = 8
                 * get prev value = 1
                 * get conductivity = 1
                 * store new value = 1
                 *
                 * Compute intensity = 14 / 10 = 1.4
                 *
                 * Max GFLOP = min (
                 *                  38.4
                 *                  25.6 * 1.4
                 * ) = 35.84 GFLOPS
                 *
                 */
                v = w * v_old +
                               ((*src)[i+1][j  ] + (*src)[i-1][j  ] +
                                (*src)[i  ][j+1] + (*src)[i  ][j-1]) * (restw * c_cdir) +
                               ((*src)[i-1][j-1] + (*src)[i-1][j+1] +
                                (*src)[i+1][j-1] + (*src)[i+1][j+1]) * (restw * c_cdiag);
                (*dst)[i][j] = v;
            }
        }
        gettimeofday(&t2, NULL);
        par = par + timeval_subtract(&t2, &t1);

        gettimeofday(&t1, NULL);
        for (i = 0; i < h -1; ++i) {
            for (j = 0; j < w -1; ++j) {
                double v_old = (*src)[i][j];
                double v = (*dst)[i][j];

                double diff = fabs(v - v_old);
                if (diff > maxdiff) maxdiff = diff;
            }
        }

        gettimeofday(&t2, NULL);
        long int diff_time = timeval_subtract(&t2, &t1);
        seq = seq + (2*diff_time/4);
        par = par + (2*diff_time/4);


        gettimeofday(&t1, NULL);
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

        gettimeofday(&t2, NULL);
        seq = seq + timeval_subtract(&t2, &t1);
    }
    double _seq = seq / 1e6;
    double _par = par / 1e6;
    double _sum = _seq + _par;
    double _f = _seq / _sum;
    printf("par %lf / seq %lf / sum %lf / f %lf / max speedup = 1/f = %lf\n", _par, _seq, _sum, _f, 1/_f);

    /* report at end in all cases */
    fill_report(p, r, h, w, dst, src, iter, &before);

    free(c);
    free(g2);
    free(g1);

}
