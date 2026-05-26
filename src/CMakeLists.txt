# ATen XPU sources

file(GLOB device_cpp "sycl/*.cpp" "sycl/*.sycl")
file(GLOB host_cpp "./*.cpp" "./*.cc")

# BMG only
set(device_cpp_xe20)
# common SYCL kernels
set(device_cpp_common)

set(SGL_KERNEL_ENABLE_FMHA_DECODE OFF CACHE BOOL "Build FMHA decode kernels" FORCE)
option(SGL_KERNEL_ENABLE_MLA_DECODE "Build MLA decode kernels" OFF)
option(SGL_KERNEL_ENABLE_MOE "Build MoE kernels" OFF)

if(SGL_KERNEL_ENABLE_FMHA_DECODE)
    add_compile_definitions(SGL_KERNEL_ENABLE_FMHA_DECODE=1)
endif()

if(SGL_KERNEL_ENABLE_MLA_DECODE)
    add_compile_definitions(SGL_KERNEL_ENABLE_MLA_DECODE=1)
endif()

if(SGL_KERNEL_ENABLE_MOE)
    add_compile_definitions(SGL_KERNEL_ENABLE_MOE=1)
endif()

foreach(file ${device_cpp})
    get_filename_component(fname ${file} NAME)

    if(NOT SGL_KERNEL_ENABLE_MLA_DECODE AND fname STREQUAL "mla_decode.cpp")
        continue()
    endif()

    if(NOT SGL_KERNEL_ENABLE_MOE AND fname MATCHES "^(MoE.*|GroupGemmXe20)\\.cpp$")
        continue()
    endif()

    if(fname MATCHES "Xe20\\.(cpp|sycl)$")
        list(APPEND device_cpp_xe20 ${file})
    else()
        list(APPEND device_cpp_common ${file})
    endif()

endforeach()

if(SGL_KERNEL_ENABLE_FMHA_DECODE)
    include(FMHADecodeXe20.cmake)
endif()
include(FMHAPrefillXe20.cmake)
include(FMHAChunkPrefillXe20.cmake)
if(SGL_KERNEL_ENABLE_MLA_DECODE)
    include(MlaDecodeXe20.cmake)
endif()

message(STATUS "BMG files: ${device_cpp_xe20}")
message(STATUS "Common files: ${device_cpp_common}")

list(APPEND ATen_XPU_CPP_SRCS ${host_cpp})
list(APPEND ATen_XPU_SYCL_COMMON ${device_cpp_common})
list(APPEND ATen_XPU_SYCL_XE20 ${device_cpp_xe20})

if(SGL_KERNEL_ENABLE_MOE)
    include(${CMAKE_CURRENT_SOURCE_DIR}/GroupGemmXe20.cmake)
endif()

set(ATen_XPU_CPP_SRCS ${ATen_XPU_CPP_SRCS} PARENT_SCOPE)
set(ATen_XPU_SYCL_COMMON ${ATen_XPU_SYCL_COMMON} PARENT_SCOPE)
set(ATen_XPU_SYCL_XE20 ${ATen_XPU_SYCL_XE20} PARENT_SCOPE)

include(BuildOnLinux.cmake)
