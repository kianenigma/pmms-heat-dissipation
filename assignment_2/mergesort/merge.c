
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <stdio.h>
#include <ctype.h>
#include <omp.h>

/* Ordering of the vector */
typedef enum Ordering {ASCENDING, DESCENDING, RANDOM} Order;

int debug = 1;

/* Sort vector v of l elements using mergesort */
void msort(int *v, long l){

}

void print_v(int *v, long l) {
  printf("\n");
  for(long i = 0; i < l; i++) {
    if(i != 0 && (i % 10 == 0)) {
      printf("\n");
    }
    printf("%d ", v[i]);
  }
  printf("\n");
}

int main(int argc, char **argv) {

  int c;
  int seed = 42;
  long length = 1e4;
  Order order = ASCENDING;
  int *vector;

  /* Read command-line options. */
  while((c = getopt(argc, argv, "adrgl:s:")) != -1) {
    switch(c) {
      case 'a':
        order = ASCENDING;
        break;
      case 'd':
        order = DESCENDING;
        break;
      case 'r':
        order = RANDOM;
        break;
      case 'l':
        length = atol(optarg);
        break;
      case 'g':
        debug = 1;
        break;
      case 's':
        seed = atoi(optarg);
	break;
      case '?':
        if(optopt == 'l' || optopt == 's') {
          fprintf(stderr, "Option -%c requires an argument.\n", optopt);
        }
        else if(isprint(optopt)) {
          fprintf(stderr, "Unknown option '-%c'.\n", optopt);
        }
        else {
          fprintf(stderr, "Unknown option character '\\x%x'.\n", optopt);
        }
        return -1;
      default:
        return -1;
      }
  }

  /* Seed such that we can always reproduce the same random vector */
  srand(seed);

  /* Allocate vector. */
  vector = (int*)malloc(length*sizeof(int));
  if(vector == NULL) {
    fprintf(stderr, "Malloc failed...\n");
    return -1;
  }

  /* Fill vector. */
  switch(order){
    case ASCENDING:
      for(long i = 0; i < length; i++) {
        vector[i] = (int)i;
      }
      break;
    case DESCENDING:
      for(long i = 0; i < length; i++) {
        vector[i] = (int)(length - i);
      } 
      break;
    case RANDOM:
      for(long i = 0; i < length; i++) {
        vector[i] = rand();
      }
      break;
  }

  if(debug) {
    print_v(vector, length);
  }

  /* Sort */
  msort(vector, length);

  if(debug) {
    print_v(vector, length);
  }

  return 0;
}

