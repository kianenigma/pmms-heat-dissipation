#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <ctype.h>
#include <pthread.h>
#include "semaphore.h"
#include <unistd.h>

int buffer_size;
pthread_attr_t attr;
int verbose = 1;

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

void *output(void *p) {
    // read buffer and semaphores
    Comparator_params *params = (Comparator_params*) p;
    pthread_t *threads = params->threads;
    sem_t *read_full = params->full;
    sem_t *read_empty = params->empty;
    int *read_buffer = params->buffer;
    int read_out = 0;

    int read_val;
    State state = INITIAL;
    int received = 0;
    int correct = 1;
    int prev_read_val = -1;

    printf("\nOutput: \n");


    /**
     * 2 different states for an output thread
     *
     * 1. INITIAL
     *
     * 2. END
     */
    while(1) {
        sem_wait(read_full);
        read_val = read_buffer[read_out];
        read_out = (read_out + 1) % buffer_size;
        sem_post(read_empty);

        if(read_val == -1) {
            // first END received
            if(state == INITIAL) {
                state = END;
                continue;
            } else {
                // second END received
                break;
            }
        }

        // print to console if verbose
        if(verbose) {
            printf("%i\n", read_val);
        }
        received++;
        if(prev_read_val > read_val) {
            correct = 0;
        }
        prev_read_val = read_val;
    }

    // print total received numbers and correctness
    printf("\nTotal received numbers: %i\n", received);
    printf("Correctness (ASC): %i\n", correct);

    // clean up
    free(read_buffer);
    sem_destroy(read_full);
    sem_destroy(read_empty);
}

int send_value(int *write_buffer, sem_t *write_empty, sem_t *write_full, int write_in, int value) {
    sem_wait(write_empty);
    //printf("Com: Put %i in buffer\n", value);
    write_buffer[write_in] = value;
    write_in = (write_in + 1) % buffer_size;
    sem_post(write_full);

    return write_in;
}

void *comparator(void *p) {
    // read buffer and semaphores
    Comparator_params *params = (Comparator_params*) p;
    pthread_t *threads = params->threads;
    sem_t *read_full = params->full;
    sem_t *read_empty = params->empty;
    int *read_buffer = params->buffer;
    int read_out = 0;

    // write buffer and semaphores
    int *write_buffer = malloc(sizeof(int) * buffer_size);
    int write_in = 0;
    sem_t write_full;
    sem_t write_empty;
    sem_init(&write_full, 0, 0);
    sem_init(&write_empty, 0, buffer_size);


    int read_val;
    int stored_number;
    State state = INITIAL;

    while(1) {
        sem_wait(read_full);
        read_val = read_buffer[read_out];
        //printf("Com: Get %i from buffer\n", read_val);
        read_out = (read_out + 1) % buffer_size;
        sem_post(read_empty);

        /**
         * 4 different states for a comparator thread
         *
         * 1. INITIAL
         *
         * 2. COMPARE_NO_THREAD
         *
         * 3. COMPARE
         *
         * 4. END
         */
        if (state == INITIAL) {
            stored_number = read_val;
            //printf("Initial state: store number %i\n", read_val);
            state = COMPARE_NO_THREAD;
        } else if(state == COMPARE_NO_THREAD) {
            // prepare thread params
            Comparator_params params_next;
            params_next.threads = threads+1;
            params_next.empty = &write_empty;
            params_next.full = &write_full;
            params_next.buffer = write_buffer;

            if (read_val == -1) {
                // create output thread
                pthread_create(threads, &attr, output, &params_next);

                // advance state to END
                state = END;

                // send end signal
                write_in = send_value(write_buffer, &write_empty, &write_full, write_in, read_val);
                // send stored_number
                write_in = send_value(write_buffer, &write_empty, &write_full, write_in, stored_number);
                continue;
            }

            // create comparator thread
            pthread_create(threads, &attr, comparator, &params_next);

            // advance state to COMPARE
            state = COMPARE;

            // compare stored_number with read_val -> send smaller to next thread
            if (stored_number > read_val) {
                write_in = send_value(write_buffer, &write_empty, &write_full, write_in, read_val);
            } else {
                write_in = send_value(write_buffer, &write_empty, &write_full, write_in, stored_number);
                stored_number = read_val;
            }
        } else if(state == COMPARE) {
            if (read_val == -1) {
                // advance state to END
                state = END;

                // send end signal
                write_in = send_value(write_buffer, &write_empty, &write_full, write_in, read_val);
                // send stored_number
                write_in = send_value(write_buffer, &write_empty, &write_full, write_in, stored_number);
                continue;
            }

            // compare stored_number with read_val -> send smaller to next thread
            if (stored_number > read_val) {
                write_in = send_value(write_buffer, &write_empty, &write_full, write_in, read_val);
            } else {
                write_in = send_value(write_buffer, &write_empty, &write_full, write_in, stored_number);
                stored_number = read_val;
            }
        } else if (state == END) {
            // send every received number including second END, then terminate
            write_in = send_value(write_buffer, &write_empty, &write_full, write_in, read_val);

            if(read_val == -1) {
                //printf("Received second end signal %i\n", read_val);
                break; // terminate
            }
        }
    }

    // clean up
    free(read_buffer);
    sem_destroy(read_full);
    sem_destroy(read_empty);
}

void *generator(void *p) {
    Generator_params *params = (Generator_params*) p;
    int length = params->length;
    pthread_t *threads = params->threads;

    // create buffer
    int *buffer = malloc(sizeof(int) * buffer_size);
    // create counter to write to buffer
    int next_in = 0;

    // create semaphores to protect buffer
    sem_t full;
    sem_t empty;

    sem_init(&full, 0, 0);
    sem_init(&empty, 0, buffer_size);

    // create first comparator thread
    Comparator_params params_next;
    params_next.threads = threads+1;
    params_next.empty = &empty;
    params_next.full = &full;
    params_next.buffer = buffer;
    pthread_create(threads, &attr, comparator, &params_next);

    // send number by number into pipeline
    for(int i = 0; i < length; i++) {
        sem_wait(&empty);
        //printf("Gen: Put %i in buffer\n", vector[i]);
        buffer[next_in] = rand();
        next_in = (next_in + 1) % buffer_size;
        sem_post(&full);
    }

    // send 2 END symbols
    for(int i = 0; i < 2; i++) {
        sem_wait(&empty);
        buffer[next_in] = -1;
        next_in = (next_in + 1) % buffer_size;
        sem_post(&full);
    }
}


int main(int argc, char **argv) {

    int c;
    int seed = 42;
    int length = 3;
    buffer_size = 1;

    while((c = getopt(argc, argv, "l:s:b:")) != -1) {
        switch(c) {
            case 'l':
                length = atoi(optarg);
                break;
            case 's':
                seed = atoi(optarg);
                break;
            case 'b':
                buffer_size = atoi(optarg);
                break;
            case '?':
                if(optopt == 'l' || optopt == 's' || optopt == 'b') {
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

    // Seed so that we can always reproduce the same random numbers
    srand(seed);

    // save all threads dynamically to this vector
    pthread_t *threads = malloc(sizeof(pthread_t) * (length+2)); // generator + length * comparator + output

    // initialize thread attributes
    pthread_attr_init(&attr);
    pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);

    // create generator thread
    Generator_params params;
    params.length = length;
    params.threads = threads+1;

    // start pipe sort
    pthread_create(threads, &attr, generator, &params);

    // join all threads (generator, comparator and output)
    for(int i = 0; i < length+2; i++){
        pthread_join(threads[i], NULL);
    }

    printf("Parameters: -b %i -l %i -s %i\n", buffer_size, length, seed);

    free(threads);
}
