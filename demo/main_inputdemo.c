#include "input.h"

int main(int argc, char **argv)
{
    struct parameters p;
    read_parameters(&p, argc, argv);

    return 0;
}
