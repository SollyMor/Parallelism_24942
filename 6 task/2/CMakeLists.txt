cmake_minimum_required(VERSION 3.16)
project(HeatEquation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Поиск OpenMP
find_package(OpenMP REQUIRED)

# Адаптируем флаги под компилятор
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native -funroll-loops -flto -DNDEBUG")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fast -mp=gpu -DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(heat_equation 2.cpp)

target_include_directories(heat_equation PRIVATE 
    ${Boost_INCLUDE_DIRS}
    ${OpenMP_CXX_INCLUDE_DIRS}
)

target_link_libraries(heat_equation PRIVATE 
    ${Boost_LIBRARIES}
    OpenMP::OpenMP_CXX
)

# LTO только для GCC/Clang
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set_property(TARGET heat_equation PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

if(MSVC)
    target_compile_definitions(heat_equation PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

message(STATUS "Boost include dirs: ${Boost_INCLUDE_DIRS}")
message(STATUS "Boost libraries:   ${Boost_LIBRARIES}")
message(STATUS "OpenMP found:      ${OpenMP_FOUND}")
message(STATUS "OpenMP version:    ${OpenMP_CXX_VERSION}")
message(STATUS "Build type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler:      ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Flags:         ${CMAKE_CXX_FLAGS}")