#ifndef OUTPUT_H
#define OUTPUT_H

#include "input.h"
#include <stddef.h>

/* final reporting */
struct results {
    size_t niter; /* effective number of iterations */
    double tmin; /* minimum temperature in last state */
    double tmax; /* maximum temperature in last state*/
    double maxdiff; /* maximum difference during last update */
    double tavg; /* average temperature */
    double time; /* compute time in seconds */
};

void report_results(const struct parameters *p, const struct results *r);

/* helper API to output the cylinder as a picture */

void begin_picture(size_t key, 
                   size_t width, size_t height,
                   double vmin, double vmax);

void draw_point(size_t x, size_t y, double value);
void end_picture(void);

#endif
