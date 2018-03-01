cmake_minimum_required(VERSION 3.8)
project(mergesort)

set(CMAKE_CXX_STANDARD 99)
SET(CMAKE_C_COMPILER /usr/local/bin/gcc-6)

set(SOURCE_FILES
        merge.c)

add_executable(mergesort ${SOURCE_FILES})
