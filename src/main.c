#include "compute.h"

int main(int argc, char **argv)
{
    struct parameters p;
    struct results r;

    read_parameters(&p, argc, argv);

    do_compute(&p, &r);

    report_results(&p, &r);

    return 0;
}
