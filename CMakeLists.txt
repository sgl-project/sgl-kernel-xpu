cmake_minimum_required(VERSION 3.19.2)
project(sgl_kernel)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)

set(ATen_XPU_SYCL_XE20)
set(ATen_XPU_SYCL_COMMON)

# Torch
find_package(Python3 COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

execute_process(
  COMMAND ${PYTHON3_EXECUTABLE}
          -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_PY_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS ${TORCH_PY_PREFIX})
list(APPEND CMAKE_PREFIX_PATH ${TORCH_PY_PREFIX}/Torch)
find_package(Torch REQUIRED)


set(SGL_OPS_XPU_ROOT ${PROJECT_SOURCE_DIR})
list(APPEND CMAKE_MODULE_PATH ${SGL_OPS_XPU_ROOT}/cmake/Modules)

set(DPCPP_SYCL_TARGET "bmg" CACHE STRING "dpcpp target device" FORCE)

include(${SGL_OPS_XPU_ROOT}/cmake/SYCL.cmake)
include(${SGL_OPS_XPU_ROOT}/cmake/BuildFlags.cmake)

option(SGL_BUILD_MEMORY_MONITOR "Track build memory and guard host OOM" ON)
set(SGL_BUILD_MEM_FILE_LIMIT_GIB "4" CACHE STRING "Per-file compile RSS threshold in GiB")
set(SGL_BUILD_MEM_GUARD_AVAIL_GIB "0.5" CACHE STRING "Kill compile when MemAvailable drops below this GiB" FORCE)
set(SGL_BUILD_MEM_GUARD_USED_PCT "99" CACHE STRING "Kill compile when system used memory reaches this percent" FORCE)

if(SGL_BUILD_MEMORY_MONITOR)
  set(SGL_BUILD_MEM_DIR ${CMAKE_BINARY_DIR}/mem_monitor)
  file(MAKE_DIRECTORY ${SGL_BUILD_MEM_DIR})
  set(SGL_BUILD_MEM_RECORDS ${SGL_BUILD_MEM_DIR}/compile_mem.jsonl)
  set(SGL_BUILD_MEM_START_TS_FILE ${SGL_BUILD_MEM_DIR}/build_start_epoch.txt)
  file(REMOVE ${SGL_BUILD_MEM_RECORDS})
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pathlib,time; pathlib.Path(r\"${SGL_BUILD_MEM_START_TS_FILE}\").write_text(str(time.time()))"
  )

  set(SGL_BUILD_MEM_LAUNCHER
    ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/tools/memory/compiler_launcher.py
    --db ${SGL_BUILD_MEM_RECORDS}
    --file-limit-gib ${SGL_BUILD_MEM_FILE_LIMIT_GIB}
    --guard-avail-gib ${SGL_BUILD_MEM_GUARD_AVAIL_GIB}
    --guard-used-pct ${SGL_BUILD_MEM_GUARD_USED_PCT}
  )

  set(CMAKE_C_COMPILER_LAUNCHER ${SGL_BUILD_MEM_LAUNCHER})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${SGL_BUILD_MEM_LAUNCHER})

  message(STATUS "Build memory monitor enabled (per-file limit=${SGL_BUILD_MEM_FILE_LIMIT_GIB}GiB, guard avail=${SGL_BUILD_MEM_GUARD_AVAIL_GIB}GiB, guard used=${SGL_BUILD_MEM_GUARD_USED_PCT}%)")
endif()

include(FetchContent)

# SYCL support in cutlass
add_compile_definitions(CUTLASS_ENABLE_SYCL)
set(CUTLASS_ENABLE_SYCL ON CACHE BOOL "Enable SYCL in the cutlass" FORCE)
set(CUTLASS_ENABLE_BENCHMARKS  OFF CACHE BOOL "Remove benchmark to avoid cmake version issue in google benchmark" FORCE)
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable headers only mode in cutlass" FORCE)

# cutlass
FetchContent_Declare(
    repo-cutlass-sycl
    GIT_REPOSITORY https://github.com/intel/sycl-tla.git
    GIT_TAG        525faea3f0f43a8aec2d21a70d44111db639a3a9
    GIT_SHALLOW    OFF
)
FetchContent_MakeAvailable(repo-cutlass-sycl)


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${repo-cutlass-sycl_SOURCE_DIR}/include
    ${repo-cutlass-sycl_SOURCE_DIR}/tools/util/include
)

add_subdirectory(${SGL_OPS_XPU_ROOT}/src)

# Install JIT kernel headers for sglang JIT compilation
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/sgl_kernel/jit_kernel
        DESTINATION sgl_kernel/include/sgl_kernel
        FILES_MATCHING PATTERN "*.hpp")

if(SGL_BUILD_MEMORY_MONITOR)
  install(CODE
    "execute_process(COMMAND \"${Python3_EXECUTABLE}\" \"${PROJECT_SOURCE_DIR}/tools/memory/report_peaks.py\" --db \"${SGL_BUILD_MEM_RECORDS}\" --limit-gib \"${SGL_BUILD_MEM_FILE_LIMIT_GIB}\" --start-epoch-file \"${SGL_BUILD_MEM_START_TS_FILE}\")")
endif()
