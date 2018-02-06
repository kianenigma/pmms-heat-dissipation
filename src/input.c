#include "input.h"
#include "fail.h"
//#include "config.h"
#include <getopt.h>
#include <stdlib.h>
#include <stdio.h>

static double *tinit = 0;
static double *conductivity = 0;

static void cleanup(void)
{
    free(tinit);
    free(conductivity);
}

static void usage(const char *pname)
{
    printf("Usage: %s [OPTION]...\n"
           "\n"
           "  -n NUM     Set cylinder height to ROWS.\n"
           "  -m NUM     Set cylinder width to COLUMNS.\n"
           "  -i NUM     Set the maximum number of iterations to NUM.\n"
           "  -k NUM     Set the reduction period to NUM.\n"
           "  -e NUM     The the convergence threshold to NUM.\n"
           "  -c FILE    Read conductivity values from FILE.\n"
           "  -t FILE    Read initial temperature values from FILE.\n"
           "  -L NUM     Coldest temperature in input/output images.\n"
           "  -H NUM     Warmest temperature in input/output images.\n"
           "  -p NUM     Number of threads to use (when applicable).\n"
           "  -r         Print a report every reduction cycle.\n"
           "  -h         Print this help.\n"
           );
    exit(0);
}

static void readpgm_float(const char *fname,
                          size_t height, size_t width, double *data,
                          double dmin, double dmax)
{
    char format[2];
    FILE *f;
    unsigned imgw, imgh, maxv, v;
    size_t i;

    printf("Reading PGM data from %s...\n", fname);

    if (!(f = fopen(fname, "r"))) die("fopen");

    fscanf(f, "%2s", format);
    if (format[0] != 'P' || format[1] != '2') die("only ASCII PGM input is supported");
    
    if (fscanf(f, "%u", &imgw) != 1 ||
        fscanf(f, "%u", &imgh) != 1 ||
        fscanf(f, "%u", &maxv) != 1) die("invalid input");
    
    if (imgw != width || imgh != height) {
        fprintf(stderr, "input data size (%ux%u) does not match cylinder size (%zux%zu)\n",
                imgw, imgh, width, height);
        die("invalid input");
    }

    for (i = 0; i < width * height; ++i)
    {
        if (fscanf(f, "%u", &v) != 1) die("invalid data");
        data[i] = dmin + (double)v * (dmax - dmin) / maxv;
    }

    fclose(f);
}

void read_parameters(struct parameters* p, int argc, char **argv)
{
    const char *conductivity_fname = 0;
    const char *tinit_fname = 0;
    int ch;

    /* set defaults */
    p->N = 150;
    p->M = 100;
    p->maxiter = 42;
    p->period = 1000;
    p->threshold = 0.1;
    p->io_tmin = -100.0;
    p->io_tmax = 100.0;
    p->nthreads = 1;
    p->printreports = 0;
    conductivity_fname = "pattern_100x150.pgm";
    tinit_fname = "pattern_100x150.pgm";

    while ((ch = getopt(argc, argv, "c:e:hH:i:k:L:m:M:n:N:p:t:r")) != -1)
    {
        switch(ch) {
        case 'c': conductivity_fname = optarg; break;
        case 't': tinit_fname = optarg; break;
        case 'i': p->maxiter = strtol(optarg, 0, 10); break;
        case 'k': p->period = strtol(optarg, 0, 10); break;
        case 'm': case 'M': p->M = strtol(optarg, 0, 10); break;
        case 'n': case 'N': p->N = strtol(optarg, 0, 10); break;
        case 'e': p->threshold = strtod(optarg, 0); break;
        case 'L': p->io_tmin = strtod(optarg, 0); break;
        case 'H': p->io_tmax = strtod(optarg, 0); break;
        case 'p': p->nthreads = strtol(optarg, 0, 10); break;
        case 'r': p->printreports = 1; break;
        case 'h': default: usage(argv[0]);
        }
    }

    printf("Parameters:\n"
           "  -n %zu # number of rows\n"
           "  -m %zu # number of columns\n"
           "  -i %zu # maximum number of iterations\n"
           "  -k %zu # reduction period\n"
           "  -e %e # convergence threshold\n"
           "  -c %s # input file for conductivity\n"
           "  -t %s # input file for initial temperatures\n"
           "  -L %e # coolest temperature in input/output\n"
           "  -H %e # highest temperature in input/output\n"
           "  -p %zu # number of threads (if applicable)\n"
           "  -r %d # print intermediate reports every reduction cycle\n",
           p->N, p->M, p->maxiter, p->period, p->threshold,
           conductivity_fname ? conductivity_fname : "(none)",
           tinit_fname ? tinit_fname : "(none)",
           p->io_tmin, p->io_tmax,
           p->nthreads, p->printreports);

    if (!p->N || !p->M) die("empty grid");

    atexit(cleanup);

    if (!(tinit = calloc(p->N * p->M, sizeof(double)))) die("calloc");
    if (tinit_fname) 
        readpgm_float(tinit_fname, p->N, p->M, tinit, p->io_tmin, p->io_tmax);
    p->tinit = tinit;

    if (!(conductivity = calloc(p->N * p->M, sizeof(double)))) die("calloc");
    if (conductivity_fname) 
        readpgm_float(conductivity_fname, p->N, p->M, conductivity, 0.0, 1.0);
    p->conductivity = conductivity;
}


 
