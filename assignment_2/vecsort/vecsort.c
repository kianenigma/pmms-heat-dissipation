
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <stdio.h>
#include <ctype.h>
#include <omp.h>
#include <string.h>
#include <sys/time.h>

/* Ordering of the vector */
typedef enum Ordering {ASCENDING, DESCENDING, RANDOM} Order;

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

void print_2d_v(int **v, long rows, long length) {
  printf("\n");
  for (long i = 0; i < rows; i++) {
    for (long j = 0; j < length; j++) {
      printf("%d \t", v[i][j]);
    }
    printf("\n");
  }
}

/**
 * Checks whether the given vector is in sorted in ascending order.
 * @param v pointer the sorted vector
 * @param l length of the vector
 * @return 0 if not sorted, 1 if sorted
 */
int check_result(int **v, long rows, long l) {
  for (long r = 0; r < rows; r++) {
    int prev = v[r][0];

    for(long i = 1; i < l; i++) {
      if(prev > v[r][i]) {
        return 0;
      }
      prev = v[r][i];
    }
  }
  return 1;

}

/**
 * Calculates and prints results of sorting. E.g.
 *  Elements     Time in s    Elements/s
 *  10000000  3.134850e-01  3.189945e+07
 *
 * @param tv1 first time value
 * @param tv2 secon time value
 * @param length number of elements sorted
 */
void print_results(struct timeval tv1, struct timeval tv2, long length, long rows, int **v) {
  double time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
                (double) (tv2.tv_sec - tv1.tv_sec);
  printf("Output (%s):\n\n %10s %13s %13s\n",
         check_result(v, rows, length) == 0 ? "incorrect" : "correct", "Elements", "Time in s", "Elements/s");
  printf("%10zu % .6e % .6e\n",
         length*rows,
         time,
         (double) length*rows / time);

}

// Left source half is  A[low:mid-1].
// Right source half is A[mid:high-1].
// Result is            B[low:high-1].
void merge(int *a, long low, long mid, long high, int *b) {
  long i = low, j = mid;

  // While there are elements in the left or right list...
  for(long k = low; k < high; k++) {
    // If left list head exists and is <= existing right list head.
    if(i < mid && (j >= high || a[i] <= a[j])) {
      b[k] = a[i];
      i++;
    } else {
      b[k] = a[j];
      j++;
    }
  }
}

void split_seq(int *b, long low, long high, int *a) {
  if(high - low < 2)
    return;

  // recursively split
  long mid = (low + high) >> 1; // divide by 2
  split_seq(a, low, mid, b); // sort left part
  split_seq(a, mid, high, b); // sort right part

  // merge from b to a
  merge(b, low, mid, high, a);
}

void vecsort_seq(int **vector, long rows, long length) {
  struct timeval tv1, tv2;

  printf("Running sequential - Rows %d Length %d\n", rows, length);
  int **b = (int**) malloc(sizeof(int) * rows * length);
  long row;

  /* initialize b */
  for (row = 0; row < rows; row++) {
    b[row] = (int*) malloc(sizeof(int) * length);
    memcpy(b[row], vector[row], length * sizeof(int));
  }

  /* start sorting one by one */
  gettimeofday(&tv1, NULL);
  for (row = 0; row < rows; row++) {
    split_seq(b[row], 0, length, vector[row]);
  }
  gettimeofday(&tv2, NULL);

  print_results(tv1, tv2, length, rows, vector);
}





void vecsort_par() {

}

int main(int argc, char **argv) {

  int c;
  int seed = 42;
  long length = 1e4;
  long rows = 1e2;
  int sequential = 0;
  int debug = 0;
  Order order = ASCENDING;

  /* Read command-line options. */
  while((c = getopt(argc, argv, "adrgl:s:R:S")) != -1) {
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
      case 'R':
        rows = atoi(optarg);
        break;
      case 'S':
        sequential = 1;
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
  int **vector = (int**)malloc(rows * length * sizeof(int));
  if (vector == NULL) {
    printf("Malloc failed...");
    return -1;
  }

  for (int r = 0; r < rows; r++) {
    /* Seed such that we can always reproduce the same random vector */
    srand(seed+r);

    /* Allocate vector. */
    int *array;
    array = (int*)malloc(length*sizeof(int));
    if(array == NULL) {
      fprintf(stderr, "Malloc failed...\n");
      return -1;
    }

    /* Fill array. */
    switch(order){
      case ASCENDING:
        for(long i = 0; i < length; i++) {
          array[i] = (int)i;
        }
        break;
      case DESCENDING:
        for(long i = 0; i < length; i++) {
          array[i] = (int)(length - i);
        }
        break;
      case RANDOM:
        for(long i = 0; i < length; i++) {
          array[i] = rand();
        }
        break;
    }

    /* Assign array to vector elems */
    vector[r] = array;
  }

  if (debug) {
    printf("Initial vector ::");
    print_2d_v(vector, rows, length);
  }

  /* Sort */
  if (sequential) {
    vecsort_seq(vector, rows, length);
  }

  if (debug) {
    printf("Final vector ::");
    print_2d_v(vector, rows, length);
  }


  return 0;
}

