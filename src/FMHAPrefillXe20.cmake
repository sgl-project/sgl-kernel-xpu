# Generate FMHA prefill kernel instantiation files.
#
# 16-bit-query FMHA kernels support bf16 and fp16 queries (each compiled into
# independent shared libraries). The generated translation units (and thus the
# resulting shared libraries) are split along these dimensions to keep peak
# compiler memory low:
#   1. paged vs non-paged (no_page) attention -> separate runner types
#      (FmhaPrefillRunner<HD> vs FmhaPrefillNpRunner<HD>);
#   2. KV-cache dtype: 16-bit (bf16/fp16) vs fp8 (e4m3/e5m2) -> separate TUs;
#   3. for the 16-bit paths, query dtype bf16 vs fp16 -> separate TUs.
# The paged and non-paged KV paths support INDEPENDENT sets of head dimensions.
# Non-paged prefill supports 16-bit KV only (no fp8 KV cache).
set(FMHA_PREFILL_PAGED_HEAD_DIMS 64 96 128 192 256 512)
set(FMHA_PREFILL_NP_HEAD_DIMS 64 72 80 96 128 192 256 512)

# Paged prefill (FmhaPrefillRunner) and non-paged prefill (FmhaPrefillNpRunner).
set(FMHA_PREFILL_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/flash_attention_v2/xe_fmha_fwd_prefill_kernel.cpp.in")

set(FMHA_PREFILL_NOPAGE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/flash_attention_v2/xe_fmha_fwd_prefill_nopage_kernel.cpp.in")

# FP8 KV-cache prefill path is split into a dedicated runner TU
# (FmhaPrefillFp8Runner) so its heavy e4m3/e5m2 kernel instantiations do not
# inflate the 16-bit-KV prefill TU's peak compiler memory. The KV cache is fp8
# while the query dtype is bf16 or fp16 (each its own TU / .so).
set(FMHA_PREFILL_FP8_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/flash_attention_v2/xe_fmha_fwd_prefill_fp8_kernel.cpp.in")

# Per-HEAD_DIM prefill tile shapes (paged and non-paged) now live in the shared
# header sycl/kernels/flash_attention_v2/fmha_tile_dispatch.h, consumed both by
# the AOT kernel templates (compile-time lookup by HEAD_DIM) and the runtime-JIT
# wrapper. Only the ENABLE_SCORE_BLOCK2D build option remains here.
option(
    FMHA_PREFILL_ENABLE_SCORE_BLOCK2D_512
    "Reuse QK scores across the two output tiles for HEAD_DIM=512 prefill"
    ON)

# --- Paged prefill + FP8: paged head dims only. ---
# prefill_page (16-bit KV) and prefill_page (fp8 KV) are independent shared
# libraries per (HEAD_DIM, query dtype, KV dtype).
#
# Generated file name order (so name = lib<file>.so):
#   xe_fmha_fwd_prefill_<page|nopage>_<HEAD_DIM>_<Qtype>_<KVtype>
# where the trailing tags are the query dtype then the KV-cache dtype (16-bit:
# KV==Q -> *_bf16_bf16 / *_fp16_fp16; fp8: KV=fp8 -> *_bf16_fp8, bf16 query only).
set(FMHA_PREFILL_ELEM_TAGS bf16 fp16)

foreach(HEAD_DIM ${FMHA_PREFILL_PAGED_HEAD_DIMS})
    # Tile-shape params (TILED_Q/KV/NUM_SG/OUT) come from the shared header
    # fmha_tile_dispatch.h, looked up in the template by HEAD_DIM at compile time.
    set(ENABLE_SCORE_BLOCK2D 0)
    if(HEAD_DIM STREQUAL "512" AND FMHA_PREFILL_ENABLE_SCORE_BLOCK2D_512)
        set(ENABLE_SCORE_BLOCK2D 1)
    endif()

    foreach(ELEM_TAG ${FMHA_PREFILL_ELEM_TAGS})
        if(ELEM_TAG STREQUAL "bf16")
            set(ELEM_TYPE "cutlass::bfloat16_t")
        else()
            set(ELEM_TYPE "cutlass::half_t")
        endif()
        set(DT16 "${ELEM_TAG}_${ELEM_TAG}")
        set(DTFP8 "${ELEM_TAG}_fp8")

        set(GENERATED_FILE
            "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_prefill_page_${HEAD_DIM}_${DT16}.cpp")
        configure_file(${FMHA_PREFILL_TEMPLATE} ${GENERATED_FILE} @ONLY)
        list(APPEND device_cpp_xe20 ${GENERATED_FILE})

        # FP8 KV cache: bf16 query only (KV dtype = fp8, Q dtype = bf16).
        # fp16 query + fp8 KV is intentionally not built.
        if(ELEM_TAG STREQUAL "bf16")
            set(GENERATED_FP8_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_prefill_page_${HEAD_DIM}_${DTFP8}.cpp")
            configure_file(${FMHA_PREFILL_FP8_TEMPLATE} ${GENERATED_FP8_FILE} @ONLY)
            list(APPEND device_cpp_xe20 ${GENERATED_FP8_FILE})
        endif()
    endforeach()
endforeach()

# --- Non-paged (no_page) prefill: np head dims only, no fp8. 16-bit KV (KV==Q). ---
foreach(HEAD_DIM ${FMHA_PREFILL_NP_HEAD_DIMS})
    # Non-paged tile-shape params come from the shared header fmha_tile_dispatch.h.
    set(ENABLE_SCORE_BLOCK2D 0)
    if(HEAD_DIM STREQUAL "512" AND FMHA_PREFILL_ENABLE_SCORE_BLOCK2D_512)
        set(ENABLE_SCORE_BLOCK2D 1)
    endif()

    foreach(ELEM_TAG ${FMHA_PREFILL_ELEM_TAGS})
        if(ELEM_TAG STREQUAL "bf16")
            set(ELEM_TYPE "cutlass::bfloat16_t")
        else()
            set(ELEM_TYPE "cutlass::half_t")
        endif()
        set(DT16 "${ELEM_TAG}_${ELEM_TAG}")

        set(GENERATED_NP_FILE
            "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_prefill_nopage_${HEAD_DIM}_${DT16}.cpp")
        configure_file(${FMHA_PREFILL_NOPAGE_TEMPLATE} ${GENERATED_NP_FILE} @ONLY)
        list(APPEND device_cpp_xe20 ${GENERATED_NP_FILE})
    endforeach()
endforeach()
