#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>
#include <stdio.h>
#include <ctype.h>
#include <omp.h>
#include <string.h>
#include <sys/time.h>

int TASK_THREADS = 6;
int DATA_THREADS = 2;


/* Ordering of the vector */
typedef enum Ordering {
    ASCENDING, DESCENDING, RANDOM
} Order;

/**
 * Prints the given vector.
 * @param v pointer to vector
 * @param l length of the vector
 */
void print_v(int *v, long l) {
    printf("\n");
    for (long i = 0; i < l; i++) {
        if (i != 0 && (i % 10 == 0)) {
            printf("\n");
        }
        printf("%d ", v[i]);
    }
    printf("\n");
}

/**
 * Print a two dim pointer vector
 * @param v vector
 * @param rows
 * @param length
 */
void print_2d_v(int **v, long rows, const long* row_lengths) {
    printf("\n");
    for (long i = 0; i < rows; i++) {
        long length = row_lengths[i];
        for (long j = 0; j < length; j++) {
            printf("%d \t", v[i][j]);
        }
        printf("\n");
    }
}

/**
 * Checks whether the given vectors is in sorted in ascending order.
 * @param v pointer to the sorted vectors
 * @param rows number of vectors
 * @param l length of the vectors
 * @return 0 if not sorted, 1 if sorted
 */
int check_result(int **v, long rows, const long* row_lengths) {
    for (long r = 0; r < rows; r++) {
        int prev = v[r][0];
        long l = row_lengths[r];
        for (long i = 1; i < l; i++) {
            if (prev > v[r][i]) {
                printf("warning: vector at row[%ld] is not sorted", r);
                print_v(v[r], l);
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
 * @param tv2 second time value
 * @param length number of elements sorted
 */
void print_results(struct timeval tv1, struct timeval tv2, long rows, long sum_elements, long* row_lengths, int **v) {
    double time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
                  (double) (tv2.tv_sec - tv1.tv_sec);
    printf("Output \033[32;1m(%s)\033[0m:\n\n %10s %13s %13s\n ",
           check_result(v, rows, row_lengths) == 0 ? "incorrect" : "correct", " Elements", "Time in s", "Elements/s");
    printf("%10zu % .6e % .6e\n",
           sum_elements,
           time,
           (double) sum_elements / time);
}

/**
 * Merges two sub-lists together.
 *
 * Left source half is  A[low:mid-1]
 * Right source half is A[mid:high-1]
 * Result is            B[low:high-1].
 *
 * @param a sublist a
 * @param low lower bound
 * @param mid mid bound
 * @param high upper bound
 * @param b sublist b
 */
void merge(const int *a, long low, long mid, long high, int *b) {
    long i = low, j = mid;

    // While there are elements in the left or right list...
    for (long k = low; k < high; k++) {
        // If left list head exists and is <= existing right list head.
        if (i < mid && (j >= high || a[i] <= a[j])) {
            b[k] = a[i];
            i++;
        } else {
            b[k] = a[j];
            j++;
        }
    }
}

/**
 * Sequentially splits the given list and merges them (sorted) together.
 *
 * @param b sublist b
 * @param low lower bound
 * @param high upper bound
 * @param a sublist a
 */
void split_seq(int *b, long low, long high, int *a) {
    if (high - low < 2)
        return;

    // recursively split
    long mid = (low + high) >> 1; // divide by 2
    split_seq(a, low, mid, b); // sort left part
    split_seq(a, mid, high, b); // sort right part

    // merge from b to a
    merge(b, low, mid, high, a);
}

/**
 * Splits in parallel the given list and merges them (sorted) together.
 *
 * @param b sublist b
 * @param low lower bound
 * @param high upper bound
 * @param a sublist a
 */
void split_parallel(int *b, long low, long high, int *a) {
    // parallelism threshold
    if (high - low < 1000) {
        split_seq(b, low, high, a);
        return;
    }

    // recursively split
    long mid = (low + high) >> 1; // divide by 2
#pragma omp task shared(a, b) firstprivate(low, mid)
    split_parallel(a, low, mid, b); // sort left part
#pragma omp task shared(a, b) firstprivate(mid, high)
    split_parallel(a, mid, high, b); // sort right part

#pragma omp taskwait

    // merge from b to a
    merge(b, low, mid, high, a);
}

/**
 * Sort the vector of vectors with data parallelism
 * @param vector
 * @param rows
 * @param length
 */
void vecsort_datapar(int **vector, long rows, long* row_lengths, long sum_elements) {
    struct timeval tv1, tv2;

    printf("Running parallel - Rows %ld \n", rows);
    printf("Number of threads for data par %d\n", DATA_THREADS);

    int **b = (int **) malloc(sizeof(int) * sum_elements);
    long row;

    /* initialize b */
    for (row = 0; row < rows; row++) {
        long length = row_lengths[row];
        b[row] = (int *) malloc(sizeof(int) * length);
        memcpy(b[row], vector[row], length * sizeof(int));
    }

    /* start sorting one by one */
    gettimeofday(&tv1, NULL);
#pragma omp parallel for num_threads(DATA_THREADS)
    for (row = 0; row < rows; row++) {
        long length = row_lengths[row];
        split_seq(b[row], 0, length, vector[row]);
    }
    gettimeofday(&tv2, NULL);

    print_results(tv1, tv2, rows, sum_elements, row_lengths, vector);
}

/**
 * Sort the vector of vectors with both data and task parallelism
 * @param vector
 * @param rows
 * @param length
 */
void vecsort_taskpar(int **vector, long rows, long* row_lengths, long sum_elements) {
    struct timeval tv1, tv2;

    printf("Running parallel - Rows %ld \n", rows);
    printf("Number of threads for data par %d\n", DATA_THREADS);
    printf("Number of threads for task par %d\n", TASK_THREADS);

    /* needed to enable nested parallelism */
    omp_set_nested(1);

    int **b = (int **) malloc(sizeof(int) * sum_elements);

    /* initialize b */
    for (long row = 0; row < rows; row++) {
        long length = row_lengths[row];
        b[row] = (int *) malloc(sizeof(int) * length);
        memcpy(b[row], vector[row], length * sizeof(int));
    }

    /* start sorting one by one */
    gettimeofday(&tv1, NULL);
#pragma omp parallel num_threads(DATA_THREADS)
    {
#pragma omp for
        for (long row = 0; row < rows; row++) {
            long length = row_lengths[row];
#pragma omp parallel num_threads(TASK_THREADS)
            {
#pragma omp single
                {
                    split_parallel(b[row], 0, length, vector[row]);
                };
            }
        }
    }

    gettimeofday(&tv2, NULL);

    print_results(tv1, tv2, rows, sum_elements, row_lengths, vector);
}

/**
 * Sort the vector of vectors sequentially
 *
 * @param vector
 * @param rows
 * @param length
 */
void vecsort_seq(int **vector, long rows, long* row_lengths, long sum_elements) {
    struct timeval tv1, tv2;

    printf("Running sequential - %ld Rows\n", rows);
    int **b = (int **) malloc(sizeof(int) * sum_elements);
    long row;

    /* initialize b */
    for (row = 0; row < rows; row++) {
        long length = row_lengths[row];
        b[row] = (int *) malloc(sizeof(int) * length);
        memcpy(b[row], vector[row], length * sizeof(int));
    }

    /* start sorting one by one */
    gettimeofday(&tv1, NULL);
    for (row = 0; row < rows; row++) {
        long length = row_lengths[row];
        split_seq(b[row], 0, length, vector[row]);
    }
    gettimeofday(&tv2, NULL);

    print_results(tv1, tv2, rows, sum_elements, row_lengths, vector);
}

/**
 *
 * usage: ./vecsort
 *
 * arguments:
 *      -a                      initial order ascending
 *      -d                      initial order descending
 *      -r                      initial order random
 *      -l {number of elements} length of each vector. if {-v} is used, this will be the upper bound of size.
 *      -R {number of rows}     number of vectors to create
 *      -v                      variable length. if enables, {-R} vectors will be created, each with size in the lange of [l/2, l].
 *      -g                      debug mode -> print vector
 *      -s {seed}               provide seed for srand
 *      -P                      run with data parallelization only.
 *      -D {data threads count} number of threads in data parallel execution
 *      -T {task threads count} number of threads in task parallel execution
 *      -S                      executes sequentially
 *
 * Examples:
 *   - a debug example with 10 x 10 dimension and random size (total elements will NTO be 100)
 *
 *
 *   - a debug example with 10 x 10 dimension and random size (total elements will be 100)
 *
 *
 *   - run a real size example with only data parallel version
 */
int main(int argc, char **argv) {

    int c;
    int seed = 42;
    long length = 1e4;
    long rows = 1e2;
    int var_length = 0;
    long *row_lengths;
    long sum_elements = 0;
    int sequential = 0;
    int datapar_only = 0;
    int debug = 0;
    Order order = ASCENDING;

    /* Read command-line options. */
    while ((c = getopt(argc, argv, "adrgl:s:R:D:T:SPv")) != -1) {
        switch (c) {
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
            case 'P':
                datapar_only = 1;
            case 'v':
                var_length = 1;
                break;
            case 'D':
                DATA_THREADS = atoi(optarg);
                break;
            case 'T':
                TASK_THREADS = atoi(optarg);
                break;
            case '?':
                if (optopt == 'l' || optopt == 's') {
                    fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                } else if (isprint(optopt)) {
                    fprintf(stderr, "Unknown option '-%c'.\n", optopt);
                } else {
                    fprintf(stderr, "Unknown option character '\\x%x'.\n", optopt);
                }
                return -1;
            default:
                return -1;
        }
    }

    row_lengths = (long*) malloc(rows * sizeof(long));

    /* create a vector that indicates the size of each sub-vector */
    if ( var_length ) {
        printf("Variable length mode.\n");
        for (int r = 0; r < rows; r++) {
            long l = rand() % (length + 1 - (length / 2)) + (length / 2);
            row_lengths[r] = l;
            sum_elements += l;
        }
    } else {
        printf("Fixed size mode.\n");
        for (int r = 0; r < rows; r++) {
            row_lengths[r] = length;
            sum_elements += length;
        }
    }

    printf("A total of %ld elements will be sorted.\n", sum_elements);


    int **vector = (int **) malloc(sum_elements * sizeof(int));
    if (vector == NULL) {
        printf("Malloc failed...");
        return -1;
    }

    for (int r = 0; r < rows; r++) {
        /* Seed such that we can always reproduce the same random vector */
        srand(seed + r);

        /* length of this array */
        long l = row_lengths[r];

        /* Allocate vector. */
        int *array;
        array = (int *) malloc(l * sizeof(int));
        if (array == NULL) {
            fprintf(stderr, "Malloc failed...\n");
            return -1;
        }

        /* Fill array. */
        switch (order) {
            case ASCENDING:
                for (long i = 0; i < l; i++) {
                    array[i] = (int) i;
                }
                break;
            case DESCENDING:
                for (long i = 0; i < l; i++) {
                    array[i] = (int) (l - i);
                }
                break;
            case RANDOM:
                for (long i = 0; i < l; i++) {
                    array[i] = rand();
                }
                break;
        }

        /* Assign array to vector elems */
        vector[r] = array;
    }

    if (debug) {
        printf("Initial vector ::");
        print_2d_v(vector, rows, row_lengths);
    }

    /* Sort */
    if (sequential) {
        vecsort_seq(vector, rows, row_lengths, sum_elements);
    } else {
        if (datapar_only) {
            vecsort_datapar(vector, rows, row_lengths, sum_elements);
        } else {
            vecsort_taskpar(vector, rows, row_lengths, sum_elements);
        }
    }

    if (debug) {
        printf("Final vector ::");
        print_2d_v(vector, rows, row_lengths);
    }


    return 0;
}

