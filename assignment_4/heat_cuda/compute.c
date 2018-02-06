#include <sys/time.h>
#include <math.h>
#include <stdlib.h>

#include "cuda_compute.h"
#include "compute.h"
/* ... */

void do_compute(const struct parameters* p, struct results *r)
{
    /* ... */

    struct timeval before, after;
    gettimeofday(&before, NULL);

    cuda_do_compute();

    gettimeofday(&after, NULL);
    r->time = (double)(after.tv_sec - before.tv_sec) + 
        (double)(after.tv_usec - before.tv_usec) / 1e6;

    /* ... */
}
