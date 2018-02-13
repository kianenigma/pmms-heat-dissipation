#include <sys/time.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <printf.h>
#include "pmmintrin.h"
#include <x86intrin.h>
#include "emmintrin.h"

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
    const unsigned int upper_bound = M-1;
    int start_index;
    int current_col;

    double current_index_val;
    double direct_neighbours_val;
    double diagonal_neighbours_val;
    double diffs[2];
    double conductivities[2];

    __m128d vec_res;
    __m128d vec_tmp;
    __m128d vec_current_index_val;
    __m128d vec_current_t_surface_val;
    __m128d vec_current_index_conductivities;
    __m128d vec_current_index_conductivities_remain;
    __m128d vec_direct_neighbours_val;
    __m128d vec_diagonal_neighbours_val;
    const __m128d abs_bitmask = _mm_castsi128_pd(_mm_set1_epi64x(0x8000000000000000)); // set sign bit according to IEEE 754
    const __m128d one = _mm_set_pd1(1);

    gettimeofday(&tv1, NULL);
    do {
        max_diff = DBL_MIN;             // reset max_diff before each iteration
        this_row_start_idx = -M;        // reset start index

        for(row = 0; row < N; row++) {
            this_row_start_idx += M;
            row_down_start_idx = this_row_start_idx+M; // can be used because of boundary halo grid points
            row_up_start_idx = this_row_start_idx-M; // can be used because of boundary halo grid points

            for(col = 1; col < upper_bound; col+=2) {
                start_index = this_row_start_idx + col;

                // calculate values at current and next index
                vec_current_t_surface_val = _mm_loadu_pd(t_surface+start_index);
                vec_current_index_conductivities = _mm_loadu_pd(p->conductivity+start_index);
                vec_current_index_val = _mm_mul_pd(vec_current_t_surface_val, vec_current_index_conductivities);

                // calculate respective cond_weight_remain
                vec_current_index_conductivities_remain = _mm_sub_pd(one, vec_current_index_conductivities);
                _mm_store_pd(conductivities, vec_current_index_conductivities_remain);

                //
                // calculate first value
                //
                current_col = col;
                col_left = current_col-1; // col == 0 ? M-1 : col-1 ;
                col_right = current_col+1; // col == M-1 ? 0 : col+1;
                cond_weight_remain = conductivities[0];


                direct_neighbours_val = cond_weight_remain * direct_neighbour_weight * (
                        t_surface[row_up_start_idx + current_col] +
                        t_surface[row_down_start_idx + current_col] +
                        t_surface[this_row_start_idx + col_left] +
                        t_surface[this_row_start_idx + col_right]
                ); // direct neighbours
                diagonal_neighbours_val = cond_weight_remain * diagonal_neighbour_weight * (
                        t_surface[row_up_start_idx + col_left] +
                        t_surface[row_up_start_idx + col_right] +
                        t_surface[row_down_start_idx + col_left] +
                        t_surface[row_down_start_idx + col_right]
                ); // diagonal neighbours

                // write to vector
                vec_direct_neighbours_val = _mm_loaddup_pd(&direct_neighbours_val);
                vec_diagonal_neighbours_val = _mm_loaddup_pd(&diagonal_neighbours_val);



                //
                // calculate second value
                //
                current_col = col+1;
                col_left = current_col-1; // col == 0 ? M-1 : col-1 ;
                col_right = current_col+1; // col == M-1 ? 0 : col+1;
                cond_weight_remain = conductivities[1];

                direct_neighbours_val = cond_weight_remain * direct_neighbour_weight * (
                        t_surface[row_up_start_idx + current_col] +
                        t_surface[row_down_start_idx + current_col] +
                        t_surface[this_row_start_idx + col_left] +
                        t_surface[this_row_start_idx + col_right]
                ); // direct neighbours
                diagonal_neighbours_val = cond_weight_remain * diagonal_neighbour_weight * (
                        t_surface[row_up_start_idx + col_left] +
                        t_surface[row_up_start_idx + col_right] +
                        t_surface[row_down_start_idx + col_left] +
                        t_surface[row_down_start_idx + col_right]
                ); // diagonal neighbours

                // write to vector
                vec_direct_neighbours_val = _mm_loadh_pd(vec_direct_neighbours_val, &direct_neighbours_val);
                vec_diagonal_neighbours_val = _mm_loadh_pd(vec_diagonal_neighbours_val, &diagonal_neighbours_val);


                // calculate temperature at given points
                vec_tmp = _mm_add_pd(vec_current_index_val, vec_direct_neighbours_val);
                vec_res = _mm_add_pd(vec_tmp, vec_diagonal_neighbours_val);

                // store result back in temp matrix
                _mm_storeu_pd(temp+start_index,vec_res);

                // calculate diff for given points
                vec_res = _mm_sub_pd(vec_current_t_surface_val, vec_res);
                // get absolute values
                vec_res = _mm_andnot_pd(abs_bitmask, vec_res);
                _mm_store_pd(diffs, vec_res);

                abs_diff = diffs[0];
                if(abs_diff > max_diff) {
                    max_diff = abs_diff;
                }
                abs_diff = diffs[1];
                if(abs_diff > max_diff) {
                    max_diff = abs_diff;
                }
            }

            //
            // do computation for col=0
            //
            col_left = M-1; //= col == 0 ? M-1 : col-1 ;
            col_right = 1; //col == M-1 ? 0 : col+1;
            index = this_row_start_idx;
            cond_weight = p->conductivity[index];
            cond_weight_remain = 1 - cond_weight;
            t_surface_index = t_surface[index]; // get only once from memory

            // calculate temperature at given point
            temp_index = cond_weight * t_surface_index
                         + cond_weight_remain * direct_neighbour_weight * (
                    t_surface[row_up_start_idx] +
                    t_surface[row_down_start_idx] +
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


            //
            // do computation for col=M-1
            //
            col = M-1;
            col_left = col-1; //= col == 0 ? M-1 : col-1 ;
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
                    t_surface[this_row_start_idx]
            ) // direct neighbours
                         + cond_weight_remain * diagonal_neighbour_weight * (
                    t_surface[row_up_start_idx + col_left] +
                    t_surface[row_up_start_idx] +
                    t_surface[row_down_start_idx + col_left] +
                    t_surface[row_down_start_idx]
            ); // diagonal neighbours


            // calculate absolute diff between new and old value
            abs_diff = fabs(t_surface_index - temp_index);
            if(abs_diff > max_diff) {
                max_diff = abs_diff;
            }
            temp[index] = temp_index;
        }

        tmp_ptr = temp;
        temp = t_surface;
        t_surface = tmp_ptr;

        niter += 1;

        // report results
        if (p->printreports == 1) {
            if (niter % p->period == 0 && niter < p->maxiter) {
                gettimeofday(&tv2, NULL);
                r->niter = niter;
                r->maxdiff = max_diff;
                r->time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
                          (double) (tv2.tv_sec - tv1.tv_sec);
                calculate_stats(p, r, t_surface);
                report_results(p, r);
            }
        }

    } while(niter < maxiter && max_diff >= threshold);
    gettimeofday(&tv2, NULL);

    r->niter = niter;
    r->maxdiff = max_diff;
    r->time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
              (double) (tv2.tv_sec - tv1.tv_sec);
    calculate_stats(p, r, t_surface);
}
