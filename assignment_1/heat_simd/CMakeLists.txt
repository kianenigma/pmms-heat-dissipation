cmake_minimum_required(VERSION 3.8)
project(heat_seq)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES
        compute.c
        fail.c
        img.c
        input.c
        main.c
        output.c)

add_executable(heat_simd ${SOURCE_FILES})

add_custom_command(OUTPUT makeall
        COMMAND make all
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )