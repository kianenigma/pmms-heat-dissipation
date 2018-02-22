#!/usr/bin/env bash
make clean && make && \
./heat_omp -n $2 -m $2 -c ./../../images/pat2_"$2"x"$2".pgm -t ./../../images/plasma_"$2"x"$2".pgm -i $3 -p $1 "$4"