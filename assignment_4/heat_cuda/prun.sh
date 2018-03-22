#!/usr/bin/env bash
make clean && make && prun -v -1 -np 1 -native '-C GTX680' ./heat_cuda \
-n $1 -m $2 \
-c ./../../images/pat2_"$1"x"$2".pgm \
-t ./../../images/plasma_"$1"x"$2".pgm \
-i $3 
