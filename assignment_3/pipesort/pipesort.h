#ifndef PIPESORT_H
#define PIPESORT_H


typedef enum {INITIAL, COMPARE_NO_THREAD, COMPARE, END} State;

typedef struct Generator_params {
    int length;
} Generator_params;

typedef struct Comparator_params {
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

/**
 * Sends given value via buffer[write_in] and with waiting on write_empty and posting on write_full.
 *
 * @param write_buffer
 * @param write_empty
 * @param write_full
 * @param write_in
 * @param value
 * @return next write_in value
 */

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
