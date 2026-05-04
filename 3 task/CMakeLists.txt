cmake_minimum_required(VERSION 3.16)
project(parallelism_task3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/O2 /W4)
else()
  add_compile_options(-O3 -march=native -mtune=native -Wall -Wextra)
endif()

add_executable(matrix_threads task1_matrix_threads/matrix_threads.cpp)
add_executable(client_server task2_client_server/client_server.cpp)
add_executable(check_results task2_client_server/check_results.cpp)

find_package(Threads REQUIRED)
target_link_libraries(matrix_threads PRIVATE Threads::Threads)
target_link_libraries(client_server PRIVATE Threads::Threads)
target_link_libraries(check_results PRIVATE Threads::Threads)
