cmake_minimum_required(VERSION 3.19.2)
project(sgl_kernel)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)

set(ATen_XPU_SYCL_XE20)
set(ATen_XPU_SYCL_XE35)
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

# DPCPP SYCL target device.
# Possible values: bmg (Xe2), cri (Xe3P)
# To change during pip install:
#   pip install -v -e . --config-settings=cmake.define.DPCPP_SYCL_TARGET=cri
set(DPCPP_SYCL_TARGET "" CACHE STRING
    "dpcpp target device (bmg | cri). Leave empty to auto-detect from device.")

include(${SGL_OPS_XPU_ROOT}/cmake/SYCL.cmake)
include(${SGL_OPS_XPU_ROOT}/cmake/BuildFlags.cmake)

if(EXISTS "/proc/meminfo")
  option(SGL_BUILD_MEMORY_MONITOR "Track build memory and guard host OOM" ON)
  set(SGL_BUILD_MEM_FILE_LIMIT_GIB "4" CACHE STRING "Per-file compile RSS threshold in GiB")
  option(SGL_BUILD_MEM_FILE_LIMIT_FATAL "Fail the build if a single file's peak RSS reaches SGL_BUILD_MEM_FILE_LIMIT_GIB" ON)
  set(SGL_BUILD_MEM_GUARD_AVAIL_GIB "0.5" CACHE STRING "Kill compile when MemAvailable drops below this GiB" FORCE)
  set(SGL_BUILD_MEM_GUARD_USED_PCT "99" CACHE STRING "Kill compile when system used memory reaches this percent" FORCE)
else()
  option(SGL_BUILD_MEMORY_MONITOR "Track build memory and guard host OOM" OFF)
endif()

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
  if(SGL_BUILD_MEM_FILE_LIMIT_FATAL)
    list(APPEND SGL_BUILD_MEM_LAUNCHER --file-limit-fatal)
  endif()

  set(CMAKE_C_COMPILER_LAUNCHER ${SGL_BUILD_MEM_LAUNCHER})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${SGL_BUILD_MEM_LAUNCHER})

  message(STATUS "Build memory monitor enabled (per-file limit=${SGL_BUILD_MEM_FILE_LIMIT_GIB}GiB fatal=${SGL_BUILD_MEM_FILE_LIMIT_FATAL}, guard avail=${SGL_BUILD_MEM_GUARD_AVAIL_GIB}GiB, guard used=${SGL_BUILD_MEM_GUARD_USED_PCT}%)")
endif()

include(FetchContent)

# SYCL support in cutlass
add_compile_definitions(CUTLASS_ENABLE_SYCL)
set(CUTLASS_ENABLE_SYCL ON CACHE BOOL "Enable SYCL in the cutlass" FORCE)
set(CUTLASS_ENABLE_BENCHMARKS  OFF CACHE BOOL "Remove benchmark to avoid cmake version issue in google benchmark" FORCE)
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable headers only mode in cutlass" FORCE)

# Set SYCL_INTEL_TARGET based on DPCPP_SYCL_TARGET:
#   Xe2  (BMG) = 20
#   Xe3P (CRI) = 35
if(DPCPP_SYCL_TARGET MATCHES "bmg")
  add_compile_definitions(SYCL_INTEL_TARGET=20)
elseif(DPCPP_SYCL_TARGET MATCHES "cri")
  add_compile_definitions(SYCL_INTEL_TARGET=35)
  add_compile_definitions(__SYCL_TARGET_INTEL_GPU_CRI__)
else()
  message(FATAL_ERROR
    "Unsupported DPCPP_SYCL_TARGET: ${DPCPP_SYCL_TARGET}. "
    "Must be one of: bmg, cri.")
endif()

# cutlass — repo and commit depend on the target architecture
if(DPCPP_SYCL_TARGET MATCHES "bmg")
  set(CUTLASS_SYCL_REPO "https://github.com/intel/sycl-tla.git")
  set(CUTLASS_SYCL_TAG  "64584484b4279b1b4184b508af445698a4a1b603")
elseif(DPCPP_SYCL_TARGET MATCHES "cri")
  set(CUTLASS_SYCL_REPO "https://github.com/intel/sycl-tla.git")
  set(CUTLASS_SYCL_TAG  "d54352dffe8daab532abd88fc6ab0e9c9fbc6d62")
else()
  message(FATAL_ERROR
      "Unsupported DPCPP_SYCL_TARGET: ${DPCPP_SYCL_TARGET}. "
      "Must be one of: bmg, cri.")
endif()

FetchContent_Declare(
    repo-cutlass-sycl
    GIT_REPOSITORY ${CUTLASS_SYCL_REPO}
    GIT_TAG        ${CUTLASS_SYCL_TAG}
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
