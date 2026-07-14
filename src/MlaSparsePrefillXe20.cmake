# Generate Sparse MLA prefill kernel instantiation files for DeepSeek V4.
# Each (D_QK, HAVE_TOPK_LENGTH, HAS_ATTN_SINK) combination is compiled as a
# separate library to parallelize and speed up compilation.

set(MLA_SPARSE_PREFILL_D_QKS 512 576)
set(MLA_SPARSE_PREFILL_TOPK_LENGTHS false true)
set(MLA_SPARSE_PREFILL_ATTN_SINKS false true)

set(MLA_SPARSE_PREFILL_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/sparse_mla_prefill_fwd_kernel.cpp.in")

foreach(D_QK ${MLA_SPARSE_PREFILL_D_QKS})
    foreach(HAVE_TOPK_LENGTH ${MLA_SPARSE_PREFILL_TOPK_LENGTHS})
        foreach(HAS_ATTN_SINK ${MLA_SPARSE_PREFILL_ATTN_SINKS})
            if(HAVE_TOPK_LENGTH)
                set(_topk_tag "topklen_")
            else()
                set(_topk_tag "")
            endif()
            if(HAS_ATTN_SINK)
                set(_sink_tag "sink")
            else()
                set(_sink_tag "nosink")
            endif()
            set(GENERATED_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/sparse_mla_prefill_fwd_${_topk_tag}k${D_QK}_${_sink_tag}.cpp")
            configure_file(${MLA_SPARSE_PREFILL_TEMPLATE} ${GENERATED_FILE} @ONLY)
            list(APPEND device_cpp_common ${GENERATED_FILE})
        endforeach()
    endforeach()
endforeach()
