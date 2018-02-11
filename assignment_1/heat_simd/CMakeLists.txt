cmake_minimum_required(VERSION 3.8)
project(heat_simd)

set(CMAKE_CXX_STANDARD 99)
SET(CMAKE_C_COMPILER /usr/local/bin/gcc-5)
include_directories(./../../include/)

set(SOURCE_FILES
        compute.c
        fail.c
        img.c
        input.c
        main.c
        output.c)

add_executable(heat_simd ${SOURCE_FILES})
