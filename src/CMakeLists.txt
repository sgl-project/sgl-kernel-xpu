# ATen XPU sources

file(GLOB device_cpp "sycl/*.cpp" "sycl/*.sycl")
file(GLOB host_cpp "./*.cpp" "./*.cc")

# BMG only
set(device_cpp_xe20)
# common SYCL kernels
set(device_cpp_common)

foreach(file ${device_cpp})
    get_filename_component(fname ${file} NAME)

    if(fname MATCHES "Xe20\\.(cpp|sycl)$")
        list(APPEND device_cpp_xe20 ${file})
    else()
        list(APPEND device_cpp_common ${file})
    endif()

endforeach()

# Fast-iteration switch: build ONLY the gemm_sqrsum kernel and skip every other
# kernel library. The host torch_extension_sycl.cc must also have its other op
# registrations disabled (see its BUILD_ONLY_GEMM_SQRSUM #if 0 block) so the
# module doesn't reference symbols from the skipped libraries. Default OFF.
option(BUILD_ONLY_GEMM_SQRSUM "Build only the gemm_sqrsum kernel (fast iteration)" OFF)

if(BUILD_ONLY_GEMM_SQRSUM)
    message(STATUS "BUILD_ONLY_GEMM_SQRSUM=ON: building only gemm_sqrsum, skipping all other kernels")
    # Keep only gemm_sqrsum.cpp from the common SYCL sources; drop the rest.
    list(FILTER device_cpp_common INCLUDE REGEX "gemm_sqrsum\\.cpp$")
    # Skip every other Xe20 source (GroupGemm/Mla/etc. are added via includes below).
    set(device_cpp_xe20)
    # Only the gemm_sqrsum generator; do NOT include FMHA/Mla/GroupGemm.
    include(GemmSqrSumXe20.cmake)
else()
    option(BUILD_FMHA "Build FMHA decode/prefill kernels" ON)
    if(BUILD_FMHA)
        add_compile_definitions(BUILD_FMHA)
        include(FMHADecodeXe20.cmake)
        include(FMHAPrefillXe20.cmake)
    else()
        message(STATUS "Skipping FMHA decode/prefill kernels (BUILD_FMHA=OFF)")
        # flash_attention.cpp depends on FmhaDecodeRunner symbols instantiated by
        # FMHADecodeXe20.cmake — exclude it to avoid undefined symbol errors.
        list(FILTER device_cpp_common EXCLUDE REGEX "flash_attention\\.cpp$")
    endif()
    include(MlaDecodeXe20.cmake)
    include(GemmSqrSumXe20.cmake)
endif()

message(STATUS "BMG files: ${device_cpp_xe20}")
message(STATUS "Common files: ${device_cpp_common}")

list(APPEND ATen_XPU_CPP_SRCS ${host_cpp})
list(APPEND ATen_XPU_SYCL_COMMON ${device_cpp_common})
list(APPEND ATen_XPU_SYCL_XE20 ${device_cpp_xe20})

if(NOT BUILD_ONLY_GEMM_SQRSUM)
    include(${CMAKE_CURRENT_SOURCE_DIR}/GroupGemmXe20.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/GroupGemmMxfp4W4A16Xe20.cmake)
endif()

set(ATen_XPU_CPP_SRCS ${ATen_XPU_CPP_SRCS} PARENT_SCOPE)
set(ATen_XPU_SYCL_COMMON ${ATen_XPU_SYCL_COMMON} PARENT_SCOPE)
set(ATen_XPU_SYCL_XE20 ${ATen_XPU_SYCL_XE20} PARENT_SCOPE)

include(BuildOnLinux.cmake)
