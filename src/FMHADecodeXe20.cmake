# Generate FMHA decode kernel instantiation files.
#
# All FMHA kernels use a bf16 query. The generated translation units (and thus
# the resulting shared libraries) are split along two dimensions to keep peak
# compiler memory low:
#   1. paged vs non-paged (no_page) attention -> separate runner types
#      (FmhaDecodeRunner<QG,HD,PS> vs FmhaDecodeNpRunner<QG,HD>);
#   2. KV-cache dtype: 16-bit (bf16) vs fp8 (e4m3/e5m2) -> separate TUs.
# The paged and non-paged KV paths support INDEPENDENT sets of head dimensions.
# Non-paged decode supports 16-bit KV only (no fp8 KV cache, no split-KV).
set(FMHA_DECODE_QG_SIZES 1 2 4 8 16)
set(FMHA_DECODE_PAGED_HEAD_DIMS 64 96 128 192 256 512)
set(FMHA_DECODE_NP_HEAD_DIMS 64 72 80 96 128 192)
set(FMHA_DECODE_PAGE_SIZES 64 128)

# Per-HEAD_DIM KV-tile size for the NON-PAGED (contiguous ragged) decode path.
# The paged decode kernel uses PAGE_SIZE as its KV tile; the non-paged path has
# no natural page size, so it gets its own KV-tile constant that can be tuned
# independently. Must be a multiple of 16. Only the head dims in
# FMHA_DECODE_NP_HEAD_DIMS need an entry here.
# Note: Larger head dimensions require smaller KV tiles to avoid running out of
# registers/local memory on Level Zero backend (UR_RESULT_ERROR_OUT_OF_RESOURCES).
set(FMHA_DECODE_TILED_KV_NP_64 512)
set(FMHA_DECODE_TILED_KV_NP_72 512)
set(FMHA_DECODE_TILED_KV_NP_80 512)
set(FMHA_DECODE_TILED_KV_NP_96 512)
set(FMHA_DECODE_TILED_KV_NP_128 512)
set(FMHA_DECODE_TILED_KV_NP_192 512)

# Paged decode (FmhaDecodeRunner) and non-paged decode (FmhaDecodeNpRunner).
set(FMHA_DECODE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/flash_attention_v2/xe_fmha_fwd_decode_kernel.cpp.in")

set(FMHA_DECODE_NOPAGE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/flash_attention_v2/xe_fmha_fwd_decode_nopage_kernel.cpp.in")

set(FMHA_SPLIT_DECODE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/flash_attention_v2/xe_fmha_fwd_split_decode_kernel.cpp.in")

# FP8 KV-cache paths are split into dedicated runner TUs (FmhaDecodeFp8Runner /
# FmhaSplitDecodeFp8Runner) so their heavy e4m3/e5m2 kernel instantiations do not
# inflate the bf16/fp16 decode / split-decode TUs' peak compiler memory.
set(FMHA_DECODE_FP8_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/flash_attention_v2/xe_fmha_fwd_decode_fp8_kernel.cpp.in")

set(FMHA_SPLIT_DECODE_FP8_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/flash_attention_v2/xe_fmha_fwd_split_decode_fp8_kernel.cpp.in")

foreach(QG_SZ ${FMHA_DECODE_QG_SIZES})
    # --- Paged decode + split-decode: paged head dims only, bf16 query only. ---
    # Each (QG, HEAD_DIM, PAGE_SIZE) yields independent shared libraries split by
    # KV-cache dtype:
    #   decode_paged / split_decode         (16-bit KV)
    #   decode_fp8   / split_decode_fp8      (e4m3/e5m2 KV)
    foreach(HEAD_DIM ${FMHA_DECODE_PAGED_HEAD_DIMS})
        foreach(PAGE_SIZE ${FMHA_DECODE_PAGE_SIZES})
            set(GENERATED_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_decode_paged_kernel_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}.cpp")
            configure_file(${FMHA_DECODE_TEMPLATE} ${GENERATED_FILE} @ONLY)
            list(APPEND device_cpp_common ${GENERATED_FILE})

            set(GENERATED_FP8_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_decode_fp8_kernel_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}.cpp")
            configure_file(${FMHA_DECODE_FP8_TEMPLATE} ${GENERATED_FP8_FILE} @ONLY)
            list(APPEND device_cpp_common ${GENERATED_FP8_FILE})

            set(GENERATED_SPLIT_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_split_decode_kernel_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}.cpp")
            configure_file(${FMHA_SPLIT_DECODE_TEMPLATE} ${GENERATED_SPLIT_FILE} @ONLY)
            list(APPEND device_cpp_common ${GENERATED_SPLIT_FILE})

            set(GENERATED_SPLIT_FP8_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_split_decode_fp8_kernel_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}.cpp")
            configure_file(${FMHA_SPLIT_DECODE_FP8_TEMPLATE} ${GENERATED_SPLIT_FP8_FILE} @ONLY)
            list(APPEND device_cpp_common ${GENERATED_SPLIT_FP8_FILE})
        endforeach()
    endforeach()

    # --- Non-paged (no_page) decode: np head dims only, no page size, no fp8, bf16 only. ---
    foreach(HEAD_DIM ${FMHA_DECODE_NP_HEAD_DIMS})
        set(TILED_KV_NP ${FMHA_DECODE_TILED_KV_NP_${HEAD_DIM}})
        if(NOT TILED_KV_NP)
            message(FATAL_ERROR "Missing non-paged KV tile (FMHA_DECODE_TILED_KV_NP_${HEAD_DIM}) for decode HEAD_DIM=${HEAD_DIM}")
        endif()

        set(GENERATED_NP_FILE
            "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_decode_nopage_kernel_${QG_SZ}_${HEAD_DIM}.cpp")
        configure_file(${FMHA_DECODE_NOPAGE_TEMPLATE} ${GENERATED_NP_FILE} @ONLY)
        list(APPEND device_cpp_common ${GENERATED_NP_FILE})
    endforeach()
endforeach()
