#include <sys/time.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include "compute.h"
#include "../../demo/output.h"


// TODO: use unsigned int wherever possible
// TODO: replace if else assignment with switch
// TODO: remove function calls in the loop
// TODO: instead of removing this function from the code, make it an inline function.

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
    const unsigned int M = (int)p->M;
    const unsigned int N = (int)p->N;
    const unsigned int maxiter = (int)p->maxiter;
    const double threshold = p->threshold;

    double *t_surface = calloc(N * M + N*2, sizeof(double));
    double *temp = calloc(N * M + N*2, sizeof(double));
    double *tmp_ptr;

    t_surface = t_surface + M; // move pointer to actual beginning of matrix
    temp = temp + M; // move pointer to actual beginning of matrix

    int row, col;

    // copy values from p->tinit
    for(row = 0; row < N; row++) {
        for(col = 0; col < M; col++) {
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
    int row_down_start_idx;             // will point to next row
    int row_up_start_idx;               // will point to prev row
    int this_row_start_idx;             // will be used as alias of row*M to prevent computation
    double cond_weight;
    double cond_weight_remain;
    int index;
    struct timeval tv1, tv2;
    double temp_index;
    double t_surface_index;
    unsigned int niter = 0;             // count iterations

    gettimeofday(&tv1, NULL);
    do {
        // reset max_diff before each iteration
        max_diff = DBL_MIN;

        for(row = 0; row < N; row++) {
            this_row_start_idx = row*M;
            row_down_start_idx = this_row_start_idx+M; // can be used because of boundary halo grid points
            row_up_start_idx = this_row_start_idx-M; // can be used because of boundary halo grid points

            // TODO: unroll loop for col == 0, col == M-1 -> prevent additional computation within loop
            for(col = 0; col < M; col++) {
                col_left = col == 0 ? M-1 : col-1 ;
                col_right = col == M-1 ? 0 : col+1;
                index = this_row_start_idx + col;
                cond_weight = p->conductivity[index];
                cond_weight_remain = 1 - cond_weight;
                t_surface_index = t_surface[index]; // get only once from memory

                // calculate temperature at given point
                temp_index = cond_weight * t_surface_index
                    + cond_weight_remain * direct_neighbour_weight * (
                        t_surface[row_up_start_idx + col] +
                        t_surface[row_down_start_idx + col] +
                        t_surface[this_row_start_idx + col_left] +
                        t_surface[this_row_start_idx + col_right]
                    ) // direct neighbours
                    + cond_weight_remain * diagonal_neighbour_weight * (
                        t_surface[row_up_start_idx + col_left] +
                        t_surface[row_up_start_idx + col_right] +
                        t_surface[row_down_start_idx + col_left] +
                        t_surface[row_down_start_idx + col_right]
                    ); // diagonal neighbours


                // calculate absolute diff between new and old value
                abs_diff = fabs(t_surface_index - temp_index);
                if(abs_diff > max_diff) {
                    max_diff = abs_diff;
                }

                temp[index] = temp_index;
            }
        }

        tmp_ptr = temp;
        temp = t_surface;
        t_surface = tmp_ptr;

        niter += 1;
    } while(niter < maxiter && max_diff >= threshold);
    gettimeofday(&tv2, NULL);

    r->niter = niter;
    r->maxdiff = max_diff;
    r->time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
         (double) (tv2.tv_sec - tv1.tv_sec);
    calculate_stats(p, r, t_surface);
}
