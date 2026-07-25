# Generate Sparse MLA prefill kernel instantiation files for DeepSeek V4.
# Each (ELEM_TAG, B_H) is compiled as a separate library to parallelize compilation.
# Mirrors MlaSparseDecodeXe20.cmake; the prefill 2-stage path reuses the decode
# 2-stage device stack (only the Stage-1 gather companion differs).

set(MLA_SPARSE_PREFILL_ELEM_TAGS half bf16)
set(MLA_SPARSE_PREFILL_ELEM_SYCL_TYPES "sycl::half" "sycl::ext::oneapi::bfloat16")

set(MLA_SPARSE_PREFILL_2STAGE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/mla_sparse_prefill_2stage_kernel.cpp.in")

# The 2-stage template generates one TU per (ELEM_TAG, D_QK, B_H, HAS_ATTN_SINK),
# mirroring the sparse MLA decode path. D_QK is the QK head dim (prefill supports
# {512, 576}) and B_H the sparse analog of page size: together they key the Stage-2
# config; HAS_ATTN_SINK selects the sink epilogue variant. One variant per object file
# bounds per-file compilation memory (avoids the build OOM guard -- one sink variant
# per file instead of both). The op dispatches dtype, then D_QK, then B_H, then the
# runtime attn_sink flag.
set(MLA_SPARSE_PREFILL_2STAGE_D_QK 512 576)
set(MLA_SPARSE_PREFILL_2STAGE_B_H 8 16 32 64)
set(MLA_SPARSE_PREFILL_2STAGE_HAS_ATTN_SINK 0 1)

list(LENGTH MLA_SPARSE_PREFILL_ELEM_TAGS _num_prefill_elems)
math(EXPR _num_prefill_elems "${_num_prefill_elems} - 1")

foreach(_idx RANGE ${_num_prefill_elems})
    list(GET MLA_SPARSE_PREFILL_ELEM_TAGS ${_idx} ELEM_TAG)
    list(GET MLA_SPARSE_PREFILL_ELEM_SYCL_TYPES ${_idx} ELEM_SYCL_TYPE)

    # Two-stage: one TU per (ELEM_TAG, D_QK, B_H, HAS_ATTN_SINK).
    foreach(D_QK ${MLA_SPARSE_PREFILL_2STAGE_D_QK})
        foreach(B_H ${MLA_SPARSE_PREFILL_2STAGE_B_H})
            foreach(HAS_ATTN_SINK ${MLA_SPARSE_PREFILL_2STAGE_HAS_ATTN_SINK})
                set(GENERATED_FILE_2STAGE
                    "${CMAKE_CURRENT_BINARY_DIR}/sycl/mla_sparse_prefill_2stage_kernel_${ELEM_TAG}_${D_QK}_${B_H}_${HAS_ATTN_SINK}.cpp")
                configure_file(${MLA_SPARSE_PREFILL_2STAGE_TEMPLATE} ${GENERATED_FILE_2STAGE} @ONLY)
                list(APPEND device_cpp_xe20 ${GENERATED_FILE_2STAGE})
            endforeach()
        endforeach()
    endforeach()
endforeach()
