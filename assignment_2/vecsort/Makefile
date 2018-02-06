CURR_DIR=$(notdir $(basename $(shell pwd)))
PRJ=$(CURR_DIR)
SRC=$(wildcard *.c)
OBJ=$(patsubst %.c,%.o,$(SRC))

CC=gcc
INCLUDES=-I../../include
ifndef DEBUG
CFLAGS=-O2 -std=c99
LIB=
else
CFLAGS=-O0 -g3 -std=c99
LIB=
endif

all: $(PRJ)

$(PRJ): $(OBJ)
	$(CC) $(CFLAGS) $(INCLUDES) $(OBJ) -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@ $(LIB)	

clean:
	-rm -f $(OBJ) $(PRJ)
