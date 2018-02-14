#ifndef PMMS_HEAT_DISSIPATION_HELPERS_H
#define PMMS_HEAT_DISSIPATION_HELPERS_H

#include "input.h"
#include "output.h"

/**
 * Gets the index in the matrix based on the row-major memory layout.
 *
 * @param p struct of parameters
 * @param row the row to get
 * @param col the column to get
 * @return position in data structure
 */
int get_array_index(const struct parameters* p, int row, int col);

/**
 * Calculates the stats avg, max and min temperature on the given matrix and saves it to r.
 *
 * @param p struct of parameters
 * @param r struct of results
 * @param t_surface the given temperature matrix
 */
void calculate_stats(const struct parameters* p, struct results *r, double *t_surface);

#endif //PMMS_HEAT_DISSIPATION_HELPERS_H
