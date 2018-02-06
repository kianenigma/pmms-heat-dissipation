#include "fail.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>

void die(const char *msg)
{
    if (errno != 0) 
        perror(msg);
    else
        fprintf(stderr, "error: %s\n", msg);
    exit(1);
}   
