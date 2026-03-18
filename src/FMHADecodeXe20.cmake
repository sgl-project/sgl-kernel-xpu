# Generate FMHA decode kernel instantiation files.
# Each (QG_SZ, HEAD_DIM, PAGE_SIZE) combination is compiled as a separate
# library to parallelize and speed up compilation.

set(FMHA_DECODE_QG_SIZES 1 2 4 8 16 32)
set(FMHA_DECODE_HEAD_DIMS 64 96 128 192)
set(FMHA_DECODE_PAGE_SIZES 32 64 128)

set(FMHA_DECODE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/xe_fmha_fwd_decode_kernel.cpp.in")

foreach(QG_SZ ${FMHA_DECODE_QG_SIZES})
    foreach(HEAD_DIM ${FMHA_DECODE_HEAD_DIMS})
        foreach(PAGE_SIZE ${FMHA_DECODE_PAGE_SIZES})
            math(EXPR NUM_SG "${PAGE_SIZE} / 16")

            set(GENERATED_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_decode_kernel_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}.cpp")
            configure_file(${FMHA_DECODE_TEMPLATE} ${GENERATED_FILE} @ONLY)
            list(APPEND device_cpp_common ${GENERATED_FILE})
        endforeach()
    endforeach()
endforeach()
