cmake_minimum_required(VERSION 3.16)
project(parallelism_task3 LANGUAGES CXX)

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

option(BUILD_LEGACY_TASK3 "Build legacy task3 binaries" ON)

if(BUILD_LEGACY_TASK3)
  add_executable(matrix_threads task1_matrix_threads/matrix_threads.cpp)
  add_executable(client_server task2_client_server/client_server.cpp)
  add_executable(check_results task2_client_server/check_results.cpp)

  find_package(Threads REQUIRED)
  target_link_libraries(matrix_threads PRIVATE Threads::Threads)
  target_link_libraries(client_server PRIVATE Threads::Threads)
  target_link_libraries(check_results PRIVATE Threads::Threads)
  if(MSVC)
    target_compile_options(matrix_threads PRIVATE $<$<CONFIG:Release>:/O2> $<$<CONFIG:RelWithDebInfo>:/O2>)
    target_compile_options(client_server PRIVATE $<$<CONFIG:Release>:/O2> $<$<CONFIG:RelWithDebInfo>:/O2>)
    target_compile_options(check_results PRIVATE $<$<CONFIG:Release>:/O2> $<$<CONFIG:RelWithDebInfo>:/O2>)
  endif()
endif()

# New lab3-like layout (main functionality, but not strict copy).
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/task1/CMakeLists.txt")
  add_subdirectory(task1)
endif()
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/task2/CMakeLists.txt")
  add_subdirectory(task2)
endif()
