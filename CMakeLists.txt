cmake_minimum_required(VERSION 3.16)
project(hiram_kernel VERSION 1.0.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(
        -Wall
        -Wextra
        -Wpedantic
        -Wstrict-prototypes
        -Wmissing-prototypes
        -Wshadow
        -O3
    )
elseif(MSVC)
    message(FATAL_ERROR "MSVC does not support native __int128_t arithmetic. Use GCC or Clang.")
endif()

# Static Library Target
add_library(hiram STATIC src/hiram_eval.c)
target_include_directories(hiram PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/hiram>
    $<INSTALL_INTERFACE:include>
)

# Standalone Test Executable Target
add_executable(hiram_standalone_test src/hiram_eval.c)
target_compile_definitions(hiram_standalone_test PRIVATE HIRAM_STANDALONE_TEST)
target_include_directories(hiram_standalone_test PRIVATE include include/hiram)

# Real-Time Latency Benchmark Target
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/benchmarks/benchmark_c_latency.c")
    add_executable(hiram_benchmark benchmarks/benchmark_c_latency.c)
    target_link_libraries(hiram_benchmark PRIVATE hiram)
    target_include_directories(hiram_benchmark PRIVATE include include/hiram)
endif()
