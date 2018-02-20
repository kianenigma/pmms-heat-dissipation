cmake_minimum_required(VERSION 3.8)
project(heat_omp)

set(CMAKE_CXX_STANDARD 99)
SET(CMAKE_C_COMPILER /usr/local/bin/gcc-7)
set (CMAKE_C_FLAGS "-O2 -fopenmp -std=c99")
include_directories(./../../include/)

set(SOURCE_FILES
        compute.c
        fail.c
        img.c
        helpers.c
        input.c
        main.c
        output.c)

add_executable(heat_omp ${SOURCE_FILES})
