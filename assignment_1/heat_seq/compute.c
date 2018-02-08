#include <sys/time.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include "compute.h"
#include "../../demo/output.h"

int get_array_index(const struct parameters* p, int row, int col) {
    return p->M * row + col;
}

void calculate_stats(const struct parameters* p, struct results *r, double *t_surface) {
    double max = DBL_MIN;
    double min = DBL_MAX;
    double avg = 0;
    int row;
    int col;
    int index;

    for(row = 0; row < p->N; row++) {
        for(col = 0; col < p->M; col++) {
            index = get_array_index(p, row, col);

            avg += t_surface[index];

            if(t_surface[index] > max) {
                max = t_surface[index];
                continue;
            }
            if(t_surface[index] < min) {
                min = t_surface[index];
                continue;
            }
        }
    }

    r->tavg = avg / (p->N * p->M);
    r->tmax = max;
    r->tmin = min;
}

void do_compute(const struct parameters* p, struct results *r)
{
    const int M = (int)p->M;
    const int N = (int)p->N;

    double *t_surface = calloc(p->N * p->M + p->N*2, sizeof(double));
    double *temp = calloc(p->N * p->M + p->N*2, sizeof(double));
    double *tmp_ptr;

    t_surface = t_surface + M; // move pointer to actual beginning of matrix
    temp = temp + M; // move pointer to actual beginning of matrix

    int row, col;

    // copy values from p->tinit
    for(row = 0; row < p->N; row++) {
        for(col = 0; col < p->M; col++) {
            int index = get_array_index(p, row, col);
            t_surface[index] = p->tinit[index];
        }
    }

    // initialize boundary halo grid point that remain fixed over the iterations
    for(col = 0; col < M; col++) {
        t_surface[get_array_index(p, -1, col)] = t_surface[get_array_index(p, 0, col)];
        t_surface[get_array_index(p, N, col)] = t_surface[get_array_index(p, N-1, col)];

        temp[get_array_index(p, -1, col)] = t_surface[get_array_index(p, 0, col)];
        temp[get_array_index(p, N, col)] = t_surface[get_array_index(p, N-1, col)];
    }

    // calculate constant weights
    const double direct_neighbour_weight = (sqrt(2.0)/(sqrt(2.0)+1.0)) / 4.0;
    const double diagonal_neighbour_weight = (1.0/(sqrt(2.0)+1.0)) / 4.0;


    // initialize variables for reuse in iterations
    double max_diff;
    double abs_diff;
    int col_left;
    int col_right;
    int row_up;
    int row_down;
    double cond_weight;
    double cond_weight_remain;
    int index;
    struct timeval  tv1, tv2;

    // reset iterations count
    r->niter = 0;

    gettimeofday(&tv1, NULL);
    do {
        // reset max_diff before each iteration
        max_diff = DBL_MIN;

        for(row = 0; row < N; row++) {
            row_up = row-1; // can be used because of boundary halo grid points
            row_down = row+1; // can be used because of boundary halo grid points

            for(col = 0; col < M; col++) {
                col_left = col == 1 ? M : col-1 ;
                col_right = col == M ? 1 : col+1;
                index = get_array_index(p, row, col);
                cond_weight = p->conductivity[index];
                cond_weight_remain = 1 - cond_weight;


                // calculate temperature at given point
                temp[index] = cond_weight * t_surface[index]
                    + cond_weight_remain * direct_neighbour_weight * (
                        t_surface[get_array_index(p, row_up,col)] + 
                        t_surface[get_array_index(p, row_down,col)] + 
                        t_surface[get_array_index(p, row,col_left)] + 
                        t_surface[get_array_index(p, row,col_right)]
                    ) // direct neighbours
                    + cond_weight_remain * diagonal_neighbour_weight * (
                        t_surface[get_array_index(p, row_up,col_left)] + 
                        t_surface[get_array_index(p, row_up,col_right)] + 
                        t_surface[get_array_index(p, row_down,col_left)] + 
                        t_surface[get_array_index(p, row_down,col_right)]
                    ); // diagonal neighbours

                // calculate absolute diff between new and old value
                abs_diff = fabs(t_surface[index] - temp[index]);
                if(abs_diff > max_diff) {
                    max_diff = abs_diff;
                }
            }
        }

        tmp_ptr = temp;
        temp = t_surface;
        t_surface = tmp_ptr;

        r->niter += 1;
    } while(r->niter < p->maxiter && max_diff >= p->threshold);
    gettimeofday(&tv2, NULL);

    r->maxdiff = max_diff;
    r->time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
         (double) (tv2.tv_sec - tv1.tv_sec);
    calculate_stats(p, r, t_surface);
}
