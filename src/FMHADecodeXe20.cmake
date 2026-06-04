# Generate FMHA decode kernel instantiation files.
# Each (QG_SZ, HEAD_DIM, PAGE_SIZE) combination is compiled as a separate
# library to parallelize and speed up compilation.

set(FMHA_DECODE_QG_SIZES 1 2 4 8 16)
set(FMHA_DECODE_HEAD_DIMS 64 72 96 128 192 256 512)
set(FMHA_DECODE_PAGE_SIZES 64 128)

# Per-HEAD_DIM KV-tile size for the NON-PAGED (contiguous ragged) decode path.
# The paged decode kernel uses PAGE_SIZE as its KV tile; the non-paged path has
# no natural page size, so it gets its own KV-tile constant that can be tuned
# independently. The non-paged kernel is only emitted in the PAGE_SIZE==128
# translation unit (see xe_fmha_fwd_decode_kernel.cpp.in) and the host routes
# the non-paged decode launch to that TU. Must be a multiple of 16.
set(FMHA_DECODE_TILED_KV_NP_64 512)
set(FMHA_DECODE_TILED_KV_NP_72 512)
set(FMHA_DECODE_TILED_KV_NP_96 512)
set(FMHA_DECODE_TILED_KV_NP_128 512)
set(FMHA_DECODE_TILED_KV_NP_192 512)
set(FMHA_DECODE_TILED_KV_NP_256 512)
set(FMHA_DECODE_TILED_KV_NP_512 512)

set(FMHA_DECODE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/xe_fmha_fwd_decode_kernel.cpp.in")

set(FMHA_SPLIT_DECODE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/xe_fmha_fwd_split_decode_kernel.cpp.in")

foreach(QG_SZ ${FMHA_DECODE_QG_SIZES})
    foreach(HEAD_DIM ${FMHA_DECODE_HEAD_DIMS})
        set(TILED_KV_NP ${FMHA_DECODE_TILED_KV_NP_${HEAD_DIM}})
        foreach(PAGE_SIZE ${FMHA_DECODE_PAGE_SIZES})
            set(GENERATED_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_decode_kernel_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}.cpp")
            configure_file(${FMHA_DECODE_TEMPLATE} ${GENERATED_FILE} @ONLY)
            list(APPEND device_cpp_common ${GENERATED_FILE})

            set(GENERATED_SPLIT_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_split_decode_kernel_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}.cpp")
            configure_file(${FMHA_SPLIT_DECODE_TEMPLATE} ${GENERATED_SPLIT_FILE} @ONLY)
            list(APPEND device_cpp_common ${GENERATED_SPLIT_FILE})
        endforeach()
    endforeach()
endforeach()
