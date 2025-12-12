cmake_minimum_required(VERSION 3.19.2)
project(sgl_kernel)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)

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

include(${SGL_OPS_XPU_ROOT}/cmake/SYCL.cmake)
include(${SGL_OPS_XPU_ROOT}/cmake/BuildFlags.cmake)

include(FetchContent)

# SYCL support in cutlass
add_compile_definitions(CUTLASS_ENABLE_SYCL)
add_compile_definitions(SYCL_INTEL_TARGET)
set(CUTLASS_ENABLE_SYCL ON CACHE BOOL "Enable SYCL in the cutlass" FORCE)
set(CUTLASS_ENABLE_BENCHMARKS  OFF CACHE BOOL "Remove benchmark to avoid cmake version issue in google benchmark" FORCE)
set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable headers only mode in cutlass" FORCE)

# cutlass
FetchContent_Declare(
    repo-cutlass-sycl
    GIT_REPOSITORY https://github.com/intel/sycl-tla.git
    GIT_TAG        5a0b7a8b7024175f223f4a47535650f317bcbbf3
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
