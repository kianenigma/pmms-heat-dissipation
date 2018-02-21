#!/usr/bin/env bash
prun -v -1 -np 1 \
export OMP_NUM_THREDS=$1 && \
make clean && make && \
./heat_omp -n $2 -m $2 -c ./../../images/pat2_"$2"x"$2".pgm -t ./../../images/plasma_"$2"x"$2".pgm -i $3