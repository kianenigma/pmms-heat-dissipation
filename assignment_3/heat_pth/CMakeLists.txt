cmake_minimum_required(VERSION 3.8)
project(heat_pth)

set(CMAKE_CXX_STANDARD 99)
SET(CMAKE_C_COMPILER /usr/local/bin/gcc-7)
set (CMAKE_C_FLAGS "-O2 -lpthread -std=c99")
include_directories(./../../include/)

set(SOURCE_FILES
        compute.c
        compute_seq.c
        pthread_barrier.c
        fail.c
        img.c
        input.c
        main.c
        output.c)

add_executable(heat_pth ${SOURCE_FILES})
