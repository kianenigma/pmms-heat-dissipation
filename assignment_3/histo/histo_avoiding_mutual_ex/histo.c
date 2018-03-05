#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "pthread.h"
#include "pthread_barrier.h"

#define PALLET_SIZE 255

typedef struct thread_args {
    unsigned int start_idx;
    unsigned int end_idx;
    unsigned int width;
    unsigned int height;
    unsigned int ***restrict img;
    unsigned int **restrict histo;
} thread_args;

/**
 * Calculates and prints results of sorting. E.g.
 *  Image size     Time in s       Pixels/s
 *  10000000       3.134850e-01    3.189945e+07
 *
 * @param tv1 first time value
 * @param tv2 second time value
 */
void print_results(struct timeval tv1, struct timeval tv2, unsigned int height, unsigned width, int correct) {
    double time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 +
                  (double) (tv2.tv_sec - tv1.tv_sec);
    printf("Output \033[32;1m(%s)\033[0m:\n%10s %13s %13s\n ",
           correct == 0 ? "incorrect" : "correct", " Image size", "Time in s", "Pixels/s");
    printf("%1d \t%.6e \t%.6e\n",
           width*height,
           time,
           (double) (width*height) / time);
}

void generate_random_image(unsigned int HEIGHT, unsigned int WIDTH, unsigned int (* restrict img)[HEIGHT][WIDTH]) {
    int i, j;
    for (i = 0; i < HEIGHT; i++) {
        for (j = 0; j < WIDTH; j++) {
            unsigned int pix_value = (unsigned int)rand() % (PALLET_SIZE+1);
            (*img)[i][j] = pix_value;
        }
    }
}

void print_image(unsigned int HEIGHT, unsigned int WIDTH, unsigned int (*restrict img)[HEIGHT][WIDTH]) {
    int i, j;
    for (i = 0; i < HEIGHT; i++) {
        for (j = 0; j < WIDTH; j++) {
            printf("%d\t", (*img)[i][j]);
        }
        printf("\n");
    }
}

void print_histogram(unsigned int (*restrict histo)[PALLET_SIZE]) {
    printf("[");
    for (int i = 0; i < PALLET_SIZE; i++) {
        printf(" %d,", (*histo)[i]);
    }
    printf("]\n");
}

inline void calculate_histo_seq(unsigned int HEIGHT, unsigned int WIDTH,
                         unsigned int (*restrict img)[HEIGHT][WIDTH],
                         unsigned int (*restrict histo)[PALLET_SIZE]) {
    int i, j;
    for (i = 0; i < HEIGHT; i++) {
        for (j = 0; j < WIDTH; j++) {
            (*histo)[(*img)[i][j]] ++;
        }
    }
}


void* thread_proc(void *param) {
    thread_args *args = (thread_args*)param;
    unsigned int lb = args->start_idx;
    unsigned int ub = args->end_idx;
    unsigned int width = args->width;
    unsigned int (*restrict img)[args->height][args->width] = args->img;
    unsigned int (*restrict histo)[args->width] = args->histo;
    unsigned int i, j;

    for (i = lb; i < ub ; i++) {
        for (j = 0; j < width; j++) {
            (*histo)[(*img)[i][j]] ++;
        }
    }

}
int main(int argc, char *argv[]){
    // TODO: read from command line
    unsigned int WIDTH = 50, HEIGHT = 50, NUM_THREADS = 10;

    struct timeval before, after;


    unsigned int (*restrict img)[HEIGHT][WIDTH] = malloc(HEIGHT * WIDTH * sizeof(unsigned int));

    /* generate the image */
    generate_random_image(HEIGHT, WIDTH, img);
//    print_image(HEIGHT, WIDTH, img);

    /* common buffer */
    unsigned int (*restrict histo)[PALLET_SIZE] = malloc(PALLET_SIZE * sizeof(unsigned int));
    unsigned int (*restrict histo_ref)[PALLET_SIZE] = malloc(PALLET_SIZE * sizeof(unsigned int));
    for (int i = 0; i < PALLET_SIZE; i++) { (*histo)[i] = 0; (*histo_ref)[i] = 0; }

    /* calculate and print ref histogram */
    calculate_histo_seq(HEIGHT, WIDTH, img, histo_ref);

    /* divide the matrix */
    unsigned int rows_assigned = 0;
    unsigned int thread_row_count[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++) thread_row_count[i] = 0;

    unsigned int thread_start_idx[NUM_THREADS];
    unsigned int thread_end_idx[NUM_THREADS];

    int turn = 0;
    while (rows_assigned < HEIGHT) {
        thread_row_count[turn%NUM_THREADS]++;
        turn++;
        rows_assigned ++;
    }
    rows_assigned = 0;
    printf("\nRow Map ::\n");
    for(int i = 0; i < NUM_THREADS; i++) {
        thread_start_idx[i] = rows_assigned;
        thread_end_idx[i] = rows_assigned + thread_row_count[i];
        rows_assigned += thread_row_count[i];

        printf("Thread %d :: %d - > %d [weight=%d]\n", i, thread_start_idx[i], thread_end_idx[i],
               thread_start_idx[i] - thread_end_idx[i]);
    }

    /* spawn threads */
    pthread_t _thread_ids[NUM_THREADS];

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
    thread_args thread_args_array[NUM_THREADS];

    for (int t = 0; t < NUM_THREADS; t++) {
        thread_args args = thread_args_array[t];
        args.start_idx = thread_start_idx[t];
        args.end_idx = thread_end_idx[t];
        args.width = WIDTH;
        args.height = HEIGHT;
        args.histo = histo;
        args.img = img;

        pthread_create(&(_thread_ids[t]), &attr, (void*) thread_proc, &args);
        if (t == 0) {
            gettimeofday(&before, NULL);
        }
    }

    for (int t = 0; t < NUM_THREADS; t++) {
        pthread_join(_thread_ids[t], NULL);
    }

    gettimeofday(&after, NULL);

    int correct = 1;
    for (int h = 0; h < PALLET_SIZE; h++) {
        if ((*histo)[h] != (*histo_ref)[h]) { correct = 0; break;}
    }

    print_results(before, after, HEIGHT, WIDTH, correct);

    free(img);
    free(histo);
    free(histo_ref);
}
