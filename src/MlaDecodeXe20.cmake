# Generate MLA decode kernel instantiation files.
# Each (ELEM_TAG, PAGE_SIZE, HAS_LSE) combination is compiled as a separate
# library to parallelize and speed up compilation.

set(MLA_DECODE_ELEM_TAGS half bf16)
set(MLA_DECODE_ELEM_SYCL_TYPES "sycl::half" "sycl::ext::oneapi::bfloat16")
set(MLA_DECODE_PAGE_SIZES 16 32 64 128)

# HAS_LSE selects whether the kernel emits the softmax log-sum-exp, mirroring
# HAS_ATTN_SINK in MlaSparseDecodeXe20.cmake. One variant per object file bounds
# per-file compilation memory: crossed with the split-KV dispatch inside runMla()
# each TU holds two kernels instead of four. flash_mla_decode() resolves the
# runtime lse.has_value() to the matching 0/1 symbol.
set(MLA_DECODE_HAS_LSE 0 1)

set(MLA_DECODE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/mla_decode_kernel.cpp.in")

list(LENGTH MLA_DECODE_ELEM_TAGS _num_elems)
math(EXPR _num_elems "${_num_elems} - 1")

foreach(_idx RANGE ${_num_elems})
    list(GET MLA_DECODE_ELEM_TAGS ${_idx} ELEM_TAG)
    list(GET MLA_DECODE_ELEM_SYCL_TYPES ${_idx} ELEM_SYCL_TYPE)

    foreach(PAGE_SIZE ${MLA_DECODE_PAGE_SIZES})
        foreach(HAS_LSE ${MLA_DECODE_HAS_LSE})
            set(GENERATED_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/mla_decode_kernel_${ELEM_TAG}_${PAGE_SIZE}_${HAS_LSE}.cpp")
            configure_file(${MLA_DECODE_TEMPLATE} ${GENERATED_FILE} @ONLY)
            list(APPEND device_cpp_xe20 ${GENERATED_FILE})
            list(APPEND device_cpp_xe35 ${GENERATED_FILE})
        endforeach()
    endforeach()
endforeach()
