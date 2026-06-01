cmake_minimum_required(VERSION 3.14)

# nvc++ must be selected before project() or OpenACC objects link with g++/ld.
if(DEFINED ENV{PGCXX})
  set(CMAKE_CXX_COMPILER "$ENV{PGCXX}" CACHE FILEPATH "C++ compiler" FORCE)
else()
  foreach(_sdk_ver 25.1 24.11 23.11)
    set(_nvcpp
        "/opt/nvidia/hpc_sdk/Linux_x86_64/${_sdk_ver}/compilers/bin/nvc++")
    if(EXISTS "${_nvcpp}")
      set(CMAKE_CXX_COMPILER "${_nvcpp}" CACHE FILEPATH "C++ compiler" FORCE)
      break()
    endif()
  endforeach()
endif()

project(heat_conduction LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost REQUIRED COMPONENTS program_options)

set(HEAT_SOURCES heat_conduction.cpp)

set(ACC_MODE "host" CACHE STRING "OpenACC device: host, multicore, gpu")
set(GPU_ARCH "cc70" CACHE STRING "NVIDIA GPU architecture, e.g. cc70 for V100")
set(HEAT_OPTIMIZED OFF CACHE BOOL "Build with --optimized as default path")
set(HEAT_MINFO OFF CACHE BOOL "Enable verbose -Minfo=all diagnostics")

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "NVHPC|PGI")
  message(
    WARNING
      "CMAKE_CXX_COMPILER is '${CMAKE_CXX_COMPILER}' (${CMAKE_CXX_COMPILER_ID}). "
      "OpenACC requires NVIDIA HPC SDK (nvc++). Set PGCXX or reconfigure, e.g.\n"
      "  cmake -S . -B build -DCMAKE_CXX_COMPILER=/opt/nvidia/hpc_sdk/.../bin/nvc++")
endif()

add_executable(heat_conduction ${HEAT_SOURCES})
target_include_directories(heat_conduction PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(heat_conduction PRIVATE Boost::program_options)
target_compile_definitions(heat_conduction PRIVATE ACC_MODE_STR="${ACC_MODE}")

set(_acc_flags -O3 -acc=${ACC_MODE} -fast)

target_compile_options(heat_conduction PRIVATE ${_acc_flags})
target_link_options(heat_conduction PRIVATE ${_acc_flags})

if(ACC_MODE STREQUAL "multicore")
  target_compile_options(heat_conduction PRIVATE -mp -Mconcur)
  target_link_options(heat_conduction PRIVATE -mp -Mconcur)
endif()

if(ACC_MODE STREQUAL "gpu")
  target_compile_options(heat_conduction PRIVATE -gpu=${GPU_ARCH})
  target_link_options(heat_conduction PRIVATE -gpu=${GPU_ARCH})
endif()

if(HEAT_MINFO)
  target_compile_options(heat_conduction PRIVATE -Minfo=all)
endif()

if(HEAT_OPTIMIZED)
  target_compile_definitions(heat_conduction PRIVATE FORCE_OPTIMIZED=1)
endif()

set_target_properties(heat_conduction PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                 ${CMAKE_BINARY_DIR}/bin)
