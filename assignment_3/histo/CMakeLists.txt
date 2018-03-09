cmake_minimum_required(VERSION 3.8)
project(histo)

set(CMAKE_CXX_STANDARD 99)
SET(CMAKE_C_COMPILER /usr/local/bin/gcc-7)
set (CMAKE_C_FLAGS "-O2 -lpthread -fgnu-tm -std=c99")
include_directories(./../../include/)

set(SOURCE_FILES
        histo_avoiding_mutual_ex/histo.c
        with_mutual_ex/histo_mutex/histo.c
        with_mutual_ex/histo_atomic/histo.c
        with_mutual_ex/histo_sw_transactional/histo.c
        with_mutual_ex/histo_semaphores/histo.c
        )

add_executable(histo ${SOURCE_FILES})
