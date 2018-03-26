#include <sys/time.h>
#include <math.h>
#include <stdlib.h>

#include "cuda_compute.h"
#include "compute.h"

void do_compute(const struct parameters* p, struct results *r)
{
    cuda_do_compute(p, r);
}
