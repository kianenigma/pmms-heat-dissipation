#!/usr/bin/env bash
make clean && make && \
./heat_pth -n $1 -m $2 -c ./../../images/pat2_"$1"x"$2".pgm -t ./../../images/plasma_"$1"x"$2".pgm -i $3 -p $4 -e 0.00001