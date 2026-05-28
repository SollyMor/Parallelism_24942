cmake_minimum_required(VERSION 3.14)
project(heat_conduction LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS program_options)

set(HEAT_SOURCES heat_conduction.cpp)

# Prefer NVIDIA HPC SDK when available
if(DEFINED ENV{PGCXX})
  set(CMAKE_CXX_COMPILER "$ENV{PGCXX}")
elseif(EXISTS "/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++")
  set(CMAKE_CXX_COMPILER
      "/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/bin/nvc++")
endif()

set(ACC_MODE "host" CACHE STRING "OpenACC device: host, multicore, gpu")
set(HEAT_OPTIMIZED OFF CACHE BOOL "Build with --optimized as default path")

add_executable(heat_conduction ${HEAT_SOURCES})
target_include_directories(heat_conduction PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(heat_conduction PRIVATE Boost::program_options)
target_compile_definitions(heat_conduction PRIVATE ACC_MODE_STR="${ACC_MODE}")

target_compile_options(
  heat_conduction
  PRIVATE -O3 -acc=${ACC_MODE} -Minfo=all -fast)

if(HEAT_OPTIMIZED)
  target_compile_definitions(heat_conduction PRIVATE FORCE_OPTIMIZED=1)
endif()

set_target_properties(heat_conduction PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                 ${CMAKE_BINARY_DIR}/bin)
