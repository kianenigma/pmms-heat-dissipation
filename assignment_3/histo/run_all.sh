#!/usr/bin/env bash

cd histo_avoiding_mutual_ex && make clean && make && cd .. && \
cd with_mutual_ex/histo_atomic && make clean && make && cd .. && \
cd histo_mutex && make clean && make && cd .. && \
cd histo_semaphores && make clean && make && cd .. && \
cd histo_sw_transactional && make clean && make && cd .. && clear \

echo "::::::::: Atomic"
cd histo_atomic && ./histo_atomic -p $1 -w $2 -h $3 -s && cd .. && \
echo "::::::::: Mutex"
cd histo_mutex && ./histo_mutex -p $1 -w $2 -h $3 -s && cd .. && \
echo "::::::::: Sem"
cd histo_semaphores && ./histo_semaphores -p $1 -w $2 -h $3 -s && cd .. && \
echo "::::::::: Stm "
cd histo_sw_transactional && ./histo_sw_transactional -p $1 -w $2 -h $3 -s && cd .. &&  \
echo "::::::::: NO Sync" && \
cd .. && cd histo_avoiding_mutual_ex && ./histo_avoiding_mutual_ex -p $1 -w $2 -h $3 -s && cd ..
