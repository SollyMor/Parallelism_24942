cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(task1_stdthread VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(task1_stdthread task1.cpp)

find_package(Threads REQUIRED)
target_link_libraries(task1_stdthread PRIVATE Threads::Threads)

if(MSVC)
  target_compile_options(task1_stdthread PRIVATE /W4 $<$<CONFIG:Release>:/O2> $<$<CONFIG:RelWithDebInfo>:/O2>)
else()
  target_compile_options(task1_stdthread PRIVATE -O2 -Wall -Wextra)
endif()

enable_testing()

add_test(
  NAME task1_full_check
  COMMAND task1_stdthread
    --sizes 500,800
    --threads 1,2,4,7,8,16,20,40
    --repeats 2
    --drop-max 0
    --out-prefix ctest_speedup_dgemv_stdthread
)