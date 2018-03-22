#!/usr/bin/env bash
make clean && make && ./heat_cuda \
-n $1 -m $2 \
-c ./../../images/pat2_"$1"x"$2".pgm \
-t ./../../images/plasma_"$1"x"$2".pgm \
-i $3 
