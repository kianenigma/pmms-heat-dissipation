#!/usr/bin/env bash
make clean && make && \
./heat_omp -n $2 -m $3 -c ./../../images/pat2_"$2"x"$3".pgm -t ./../../images/plasma_"$2"x"$3".pgm -i $4 -p $1 "$5"