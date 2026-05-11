cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(task2_client_server VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(task2_client_server main.cpp)
add_executable(task2_verify verify_results.cpp)
add_executable(task2_benchmark benchmark.cpp)

if(MSVC)
  target_compile_options(task2_client_server PRIVATE /W4 $<$<CONFIG:Release>:/O2> $<$<CONFIG:RelWithDebInfo>:/O2>)
  target_compile_options(task2_verify PRIVATE /W4 $<$<CONFIG:Release>:/O2> $<$<CONFIG:RelWithDebInfo>:/O2>)
  target_compile_options(task2_benchmark PRIVATE /W4 $<$<CONFIG:Release>:/O2> $<$<CONFIG:RelWithDebInfo>:/O2>)
else()
  target_compile_options(task2_client_server PRIVATE -O2 -pthread)
  target_compile_options(task2_verify PRIVATE -O2)
  target_compile_options(task2_benchmark PRIVATE -O2 -pthread)
endif()

enable_testing()
add_test(NAME task2_generate COMMAND task2_client_server 100 promise-u)
add_test(NAME task2_verify_files COMMAND task2_verify)
set_tests_properties(task2_verify_files PROPERTIES DEPENDS task2_generate)
add_test(NAME task2_benchmark_quick COMMAND task2_benchmark 50 2 task2_benchmark_results.csv)

