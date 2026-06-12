# Generate GEMM + Square Sum kernel instantiation files.
# Each (ELEM_TAG, TILE_SIZE) combination is compiled as a separate
# library to parallelize and speed up compilation.

# Only half and bf16 are supported - float x float DPAS doesn't exist on XE
set(GEMM_SQRSUM_ELEM_TAGS half bf16)
set(GEMM_SQRSUM_ELEM_SYCL_TYPES "sycl::half" "sycl::ext::oneapi::bfloat16")

# Tile sizes: M x N x K
set(GEMM_SQRSUM_TILE_M 256)
set(GEMM_SQRSUM_TILE_N 256)
set(GEMM_SQRSUM_TILE_K 16)

set(GEMM_SQRSUM_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/gemm_sqrsum_kernel.cpp.in")

list(LENGTH GEMM_SQRSUM_ELEM_TAGS _num_elems)
math(EXPR _num_elems "${_num_elems} - 1")

foreach(_idx RANGE ${_num_elems})
    list(GET GEMM_SQRSUM_ELEM_TAGS ${_idx} ELEM_TAG)
    list(GET GEMM_SQRSUM_ELEM_SYCL_TYPES ${_idx} ELEM_SYCL_TYPE)

    set(TILE_M ${GEMM_SQRSUM_TILE_M})
    set(TILE_N ${GEMM_SQRSUM_TILE_N})
    set(TILE_K ${GEMM_SQRSUM_TILE_K})

    set(GENERATED_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/sycl/gemm_sqrsum_kernel_${ELEM_TAG}_${TILE_M}x${TILE_N}x${TILE_K}.cpp")
    configure_file(${GEMM_SQRSUM_TEMPLATE} ${GENERATED_FILE} @ONLY)
    list(APPEND device_cpp_common ${GENERATED_FILE})
endforeach()
