cmake_minimum_required(VERSION 3.8)
project(heat_seq)

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

add_executable(heat_seq ${SOURCE_FILES})

add_custom_command(OUTPUT makeall
        COMMAND make all
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )