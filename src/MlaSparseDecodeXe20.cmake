# Generate Sparse MLA fp8 decode kernel instantiation files for DeepSeek V4.
# Each (D_QK, IS_FP8_QUERY) combination is compiled as a
# separate library to parallelize and speed up compilation.

set(MLA_SPARSE_DECODE_D_QKS 512)
set(MLA_SPARSE_DECODE_FP8_QUERIES false true)

set(MLA_SPARSE_DECODE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/sparse_mla_decode_fp8_fwd_kernel.cpp.in")

foreach(D_QK ${MLA_SPARSE_DECODE_D_QKS})
    foreach(IS_FP8_QUERY ${MLA_SPARSE_DECODE_FP8_QUERIES})
        if(IS_FP8_QUERY)
            set(_q_tag "qfp8")
        else()
            set(_q_tag "qbf16")
        endif()
        set(GENERATED_FILE
            "${CMAKE_CURRENT_BINARY_DIR}/sycl/sparse_mla_decode_fp8_fwd_k${D_QK}_${_q_tag}.cpp")
        configure_file(${MLA_SPARSE_DECODE_TEMPLATE} ${GENERATED_FILE} @ONLY)
        list(APPEND device_cpp_xe20 ${GENERATED_FILE})
    endforeach()
endforeach()
