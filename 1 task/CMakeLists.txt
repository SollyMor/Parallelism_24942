cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(my-prog VERSION 0.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_link_libraries(book_exec PUBLIC book)