# Register Sparse MLA fp8 decode kernel instantiation files for DeepSeek V4.
# Each instantiation is compiled as a separate library to parallelize compilation.
# Ported from xattention (csrc/flash_attn_xpu/mla/instantiations).

set(MLA_SPARSE_DECODE_INSTANTIATIONS
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/mla_sparse_xattn/sparse_mla_decode_fp8_fwd_k512_qbf16.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/mla_sparse_xattn/sparse_mla_decode_fp8_fwd_k512_qfp8.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/mla_sparse_xattn/sparse_mla_decode_fp8_fwd_topklen_k512_qbf16.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/mla_sparse_xattn/sparse_mla_decode_fp8_fwd_topklen_k512_qfp8.cpp"
)

list(APPEND device_cpp_common ${MLA_SPARSE_DECODE_INSTANTIATIONS})
