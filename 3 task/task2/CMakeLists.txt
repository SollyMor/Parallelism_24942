cmake_minimum_required(VERSION 3.16)
project(task2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-O3 -march=native -mtune=native -Wall -Wextra)
endif()

add_executable(client_server client_server.cpp)
add_executable(check_results check_results.cpp)

find_package(Threads REQUIRED)
target_link_libraries(client_server PRIVATE Threads::Threads)
target_link_libraries(check_results PRIVATE Threads::Threads)