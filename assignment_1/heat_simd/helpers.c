#include <float.h>
#include "helpers.h"


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