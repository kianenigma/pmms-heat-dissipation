#ifndef PIPESORT_H
#define PIPESORT_H


typedef enum {INITIAL, COMPARE_NO_THREAD, COMPARE, END} State;

typedef struct Generator_params {
    int length;
    pthread_t *threads;
} Generator_params;

typedef struct Comparator_params {
    pthread_t *threads;
    int *buffer;
    sem_t *full;
    sem_t *empty;
} Comparator_params;

/**
 * Routine for an output thread.
 * Receives numbers and prints them to standard out.
 *
 * @param p void pointer to a Comparator_params struct
 */
void *output(void *p);

int send_value(int *write_buffer, sem_t *write_empty, sem_t *write_full, int write_in, int value);

/**
 * Routine for a comparator thread.
 * Receives numbers and forwards the bigger one to the next comparator thread.
 *
 * @param p void pointer to a Comparator_params struct
 */
void *comparator(void *p);

/**
 * Routine for a generator thread.
 * Generates length random numbers and passes them to a comparator thread.
 *
 * @param p void pointer to a Generator_params struct
 */
void *generator(void *p);

#endif
