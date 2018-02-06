#ifndef INPUT_H_H
#define INPUT_H_H

#include <stddef.h>

/* input parameters for the program */
struct parameters {
    /* matrix size: N rows, M columns */
    size_t N, M;

    /* maximum number of iterations */
    size_t maxiter;

    /* number of iterations for the periodic reduction */
    size_t period;

    /* print a report every reduction cycle */
    size_t printreports;

    /* convergence threshold */
    double threshold; 

    /* initial temperature, in row-major order */
    const double *tinit;

    /* conductivity values for the cylinder, in row-major order */
    const double *conductivity;

    /* temperature range in input and output images */
    double io_tmin;
    double io_tmax;

    /* number of threads */
    size_t nthreads;
};
 
void read_parameters(struct parameters* p, int argc, char **argv);

#endif
