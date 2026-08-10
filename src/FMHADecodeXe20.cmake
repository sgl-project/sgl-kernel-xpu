# Generate FMHA decode kernel instantiation files.
#
# 16-bit-query FMHA kernels support bf16 and fp16 queries (each compiled into
# independent shared libraries). The generated translation units (and thus the
# resulting shared libraries) are split along these dimensions to keep peak
# compiler memory low:
#   1. paged vs non-paged (no_page) attention -> separate runner types
#      (FmhaDecodeRunner<QG,HD,PS> vs FmhaDecodeNpRunner<QG,HD>);
#   2. KV-cache dtype: 16-bit (bf16/fp16) vs fp8 (e4m3/e5m2) -> separate TUs;
#   3. for the 16-bit paths, query dtype bf16 vs fp16 -> separate TUs.
# The paged and non-paged KV paths support INDEPENDENT sets of head dimensions.
# Non-paged decode supports 16-bit KV only (no fp8 KV cache, no split-KV).
set(FMHA_DECODE_QG_SIZES 1 2 4 8 16)
set(FMHA_DECODE_PAGED_HEAD_DIMS 64 96 128 192 256 512)
set(FMHA_DECODE_NP_HEAD_DIMS 64 72 80 96 128 192 256 512)
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
set(FMHA_DECODE_TILED_KV_NP_256 128)
set(FMHA_DECODE_TILED_KV_NP_512 128)

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

# 16-bit query element tags. Each tag produces INDEPENDENT shared libraries for
# the 16-bit-KV decode / split-decode / non-paged paths so bf16 and fp16 do not
# share a translation unit. bf16 keeps the historical (untagged) file names; fp16
# gets a `_fp16` suffix. ELEM_TYPE selects the cutlass query/KV/out element type.
set(FMHA_DECODE_ELEM_TAGS bf16 fp16)

foreach(QG_SZ ${FMHA_DECODE_QG_SIZES})
    # --- Paged decode + split-decode: paged head dims only. ---
    # Each (QG, HEAD_DIM, PAGE_SIZE) yields independent shared libraries split by
    # KV-cache dtype:
    #   decode_paged / split_decode         (16-bit KV: bf16 + fp16)
    #   decode_fp8   / split_decode_fp8      (e4m3/e5m2 KV, bf16 query)
    foreach(HEAD_DIM ${FMHA_DECODE_PAGED_HEAD_DIMS})
        foreach(PAGE_SIZE ${FMHA_DECODE_PAGE_SIZES})
            foreach(ELEM_TAG ${FMHA_DECODE_ELEM_TAGS})
                if(ELEM_TAG STREQUAL "bf16")
                    set(ELEM_TYPE "cutlass::bfloat16_t")
                    set(ELEM_SUFFIX "")
                else()
                    set(ELEM_TYPE "cutlass::half_t")
                    set(ELEM_SUFFIX "_${ELEM_TAG}")
                endif()

                set(GENERATED_FILE
                    "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_decode_paged_kernel_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}${ELEM_SUFFIX}.cpp")
                configure_file(${FMHA_DECODE_TEMPLATE} ${GENERATED_FILE} @ONLY)
                list(APPEND device_cpp_xe20 ${GENERATED_FILE})

                set(GENERATED_SPLIT_FILE
                    "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_split_decode_kernel_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}${ELEM_SUFFIX}.cpp")
                configure_file(${FMHA_SPLIT_DECODE_TEMPLATE} ${GENERATED_SPLIT_FILE} @ONLY)
                list(APPEND device_cpp_xe20 ${GENERATED_SPLIT_FILE})
            endforeach()

            set(GENERATED_FP8_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_decode_fp8_kernel_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}.cpp")
            configure_file(${FMHA_DECODE_FP8_TEMPLATE} ${GENERATED_FP8_FILE} @ONLY)
            list(APPEND device_cpp_xe20 ${GENERATED_FP8_FILE})

            set(GENERATED_SPLIT_FP8_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_split_decode_fp8_kernel_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}.cpp")
            configure_file(${FMHA_SPLIT_DECODE_FP8_TEMPLATE} ${GENERATED_SPLIT_FP8_FILE} @ONLY)
            list(APPEND device_cpp_xe20 ${GENERATED_SPLIT_FP8_FILE})
        endforeach()
    endforeach()

    # --- Non-paged (no_page) decode: np head dims only, no page size, no fp8. ---
    # 16-bit KV only (bf16 + fp16), no split-KV.
    foreach(HEAD_DIM ${FMHA_DECODE_NP_HEAD_DIMS})
        set(TILED_KV_NP ${FMHA_DECODE_TILED_KV_NP_${HEAD_DIM}})
        if(NOT TILED_KV_NP)
            message(FATAL_ERROR "Missing non-paged KV tile (FMHA_DECODE_TILED_KV_NP_${HEAD_DIM}) for decode HEAD_DIM=${HEAD_DIM}")
        endif()

        foreach(ELEM_TAG ${FMHA_DECODE_ELEM_TAGS})
            if(ELEM_TAG STREQUAL "bf16")
                set(ELEM_TYPE "cutlass::bfloat16_t")
                set(ELEM_SUFFIX "")
            else()
                set(ELEM_TYPE "cutlass::half_t")
                set(ELEM_SUFFIX "_${ELEM_TAG}")
            endif()

            set(GENERATED_NP_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_decode_nopage_kernel_${QG_SZ}_${HEAD_DIM}${ELEM_SUFFIX}.cpp")
            configure_file(${FMHA_DECODE_NOPAGE_TEMPLATE} ${GENERATED_NP_FILE} @ONLY)
            list(APPEND device_cpp_xe20 ${GENERATED_NP_FILE})
        endforeach()
    endforeach()
endforeach()
