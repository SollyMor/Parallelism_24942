cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(my-prog VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_DOUBLE "Use double precision (float if OFF)" OFF)

if(USE_DOUBLE)
    add_compile_definitions(USE_DOUBLE_TYPE)
    message(STATUS "Using double")
else()
    message(STATUS "Using float")
endif()

add_executable(sin 1.cpp)

target_compile_options(sin PRIVATE -O3 -march=native)