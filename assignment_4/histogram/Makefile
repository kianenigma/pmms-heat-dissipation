CURR_DIR=$(notdir $(basename $(shell pwd)))
PRJ=$(CURR_DIR)
SRC=$(wildcard *.c)
SRC_CUDA=$(wildcard *.cu)
OBJ=$(patsubst %.c,%.o,$(SRC))
OBJ_CUDA=$(patsubst %.cu,%.o,$(SRC_CUDA))

CC=gcc
NVCC=nvcc
INCLUDES=-I../../include

ifndef DEBUG
CFLAGS=-O2 -std=c99
LIB=
NVCCFLAGS=-O2
else
CFLAGS=-O0 -g3 -std=c99
LIB=
NVCCFLAGS=-O0 -g3
endif

all: $(PRJ)

$(PRJ): $(OBJ) $(OBJ_CUDA)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(OBJ_CUDA) $(OBJ)  -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@ $(LIB)	

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	-rm -f $(OBJ) $(OBJ_CUDA) $(PRJ)
