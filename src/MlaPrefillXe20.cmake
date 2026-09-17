# Generate MLA prefill kernel instantiation files.
# Each (ELEM_TAG, PAGE_SIZE, HAS_LSE) combination is compiled as a separate
# library to parallelize and speed up compilation.

set(MLA_PREFILL_ELEM_TAGS half bf16)
set(MLA_PREFILL_ELEM_SYCL_TYPES "sycl::half" "sycl::ext::oneapi::bfloat16")
set(MLA_PREFILL_PAGE_SIZES 16 32 64 128)

# HAS_LSE selects whether the kernels emit the softmax log-sum-exp, mirroring
# HAS_ATTN_SINK in MlaSparseDecodeXe20.cmake. One variant per object file bounds
# per-file compilation memory: three kernels per TU (one per Q-tile bucket)
# instead of six. flash_mla_prefill() resolves the runtime lse.has_value() to the
# matching 0/1 symbol.
set(MLA_PREFILL_HAS_LSE 0 1)

set(MLA_PREFILL_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/mla_prefill_kernel.cpp.in")

list(LENGTH MLA_PREFILL_ELEM_TAGS _num_elems)
math(EXPR _num_elems "${_num_elems} - 1")

foreach(_idx RANGE ${_num_elems})
    list(GET MLA_PREFILL_ELEM_TAGS ${_idx} ELEM_TAG)
    list(GET MLA_PREFILL_ELEM_SYCL_TYPES ${_idx} ELEM_SYCL_TYPE)

    foreach(PAGE_SIZE ${MLA_PREFILL_PAGE_SIZES})
        foreach(HAS_LSE ${MLA_PREFILL_HAS_LSE})
            set(GENERATED_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/mla_prefill_kernel_${ELEM_TAG}_${PAGE_SIZE}_${HAS_LSE}.cpp")
            configure_file(${MLA_PREFILL_TEMPLATE} ${GENERATED_FILE} @ONLY)
            list(APPEND device_cpp_xe20 ${GENERATED_FILE})
            list(APPEND device_cpp_xe35 ${GENERATED_FILE})
        endforeach()
    endforeach()
endforeach()
