cmake_minimum_required(VERSION 3.8)
project(histo)

set(CMAKE_CXX_STANDARD 99)
SET(CMAKE_C_COMPILER /usr/local/bin/gcc-7)
set (CMAKE_C_FLAGS "-O2 -lpthread -std=c99")
include_directories(./../../include/)

set(SOURCE_FILES
        histo_avoiding_mutual_ex/histo.c)

add_executable(histo ${SOURCE_FILES})
