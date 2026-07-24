# Generate Sparse MLA decode kernel instantiation files for DeepSeek V4.
# Each ELEM_TAG is compiled as a separate library to parallelize compilation.

set(MLA_SPARSE_DECODE_ELEM_TAGS half bf16)
set(MLA_SPARSE_DECODE_ELEM_SYCL_TYPES "sycl::half" "sycl::ext::oneapi::bfloat16")

set(MLA_SPARSE_DECODE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/mla_sparse_decode_kernel.cpp.in")

list(LENGTH MLA_SPARSE_DECODE_ELEM_TAGS _num_elems)
math(EXPR _num_elems "${_num_elems} - 1")

foreach(_idx RANGE ${_num_elems})
    list(GET MLA_SPARSE_DECODE_ELEM_TAGS ${_idx} ELEM_TAG)
    list(GET MLA_SPARSE_DECODE_ELEM_SYCL_TYPES ${_idx} ELEM_SYCL_TYPE)

    set(GENERATED_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/sycl/mla_sparse_decode_kernel_${ELEM_TAG}_128.cpp")
    configure_file(${MLA_SPARSE_DECODE_TEMPLATE} ${GENERATED_FILE} @ONLY)
    list(APPEND device_cpp_xe20 ${GENERATED_FILE})
endforeach()
