TARGETS=main_imgdemo  main_inputdemo  main_outputdemo
SRC=$(wildcard *.c)
OBJ=$(patsubst %.c,%.o,$(SRC))
OBJ_NO_MAIN=$(filter-out main_%,$(OBJ))
CC=gcc

ifdef DEBUG
CFLAGS=-O2
LIB=
else
CFLAGS=-O2 -g3
LIB=
endif

all: $(TARGETS) 

$(TARGETS): % :$(OBJ_NO_MAIN) %.o
	$(CC) $(CFLAGS) $+ -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LIB)	

clean:
	-rm -f $(OBJ) $(TARGETS) 
