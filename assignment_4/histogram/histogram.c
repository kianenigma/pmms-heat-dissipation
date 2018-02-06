#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>
#include "histo_kernel.h"

#define seed 5

int main(int argc, char **argv) {
  int length = 1024;
  srand(seed);

  /* Allocate vector. */
  int *vector = (int*)malloc(length*sizeof(int));
  if(vector == NULL) {
    fprintf(stderr, "Malloc failed...\n");
    return -1;
  }

  /* Fill vector. */
  for(long i = 0; i < length; i++) {
        vector[i] = (int)i;
  }

  histogram(vector, length);

  free(vector);

  return 0;
}

