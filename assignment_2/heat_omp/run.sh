#!/usr/bin/env bash
make clean && make && ./heat_omp \
-n $1 -m $2 \
-c ./../../images/pat2_"$1"x"$2".pgm \
-t ./../../images/plasma_"$1"x"$2".pgm \
-e 0.0001 -i $3 -p 8 \
#-k 10 -r
