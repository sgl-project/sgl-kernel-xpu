# Register Sparse MLA prefill kernel instantiation files for DeepSeek V4.
# Each instantiation is compiled as a separate library to parallelize compilation.
# Ported from xattention (csrc/flash_attn_xpu/mla/instantiations).

set(MLA_SPARSE_PREFILL_INSTANTIATIONS
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/mla_sparse_xattn/sparse_mla_prefill_fwd_k512.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/mla_sparse_xattn/sparse_mla_prefill_fwd_k576.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/mla_sparse_xattn/sparse_mla_prefill_fwd_topklen_k512.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/mla_sparse_xattn/sparse_mla_prefill_fwd_topklen_k576.cpp"
)

list(APPEND device_cpp_common ${MLA_SPARSE_PREFILL_INSTANTIATIONS})
