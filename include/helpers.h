#ifndef PMMS_HEAT_DISSIPATION_HELPERS_H
#define PMMS_HEAT_DISSIPATION_HELPERS_H

#include "input.h"
#include "output.h"

int get_array_index(const struct parameters* p, int row, int col);

void calculate_stats(const struct parameters* p, struct results *r, double *t_surface);

#endif //PMMS_HEAT_DISSIPATION_HELPERS_H
