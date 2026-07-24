# Generate FMHA prefill kernel instantiation files.
#
# All FMHA kernels use a bf16 query. The generated translation units (and thus
# the resulting shared libraries) are split along two dimensions to keep peak
# compiler memory low:
#   1. paged vs non-paged (no_page) attention -> separate runner types
#      (FmhaPrefillRunner<HD> vs FmhaPrefillNpRunner<HD>);
#   2. KV-cache dtype: 16-bit (bf16) vs fp8 (e4m3/e5m2) -> separate TUs.
# The paged and non-paged KV paths support INDEPENDENT sets of head dimensions.
# Non-paged prefill supports 16-bit KV only (no fp8 KV cache).
set(FMHA_PREFILL_PAGED_HEAD_DIMS 64 96 128 192 256 512)
set(FMHA_PREFILL_NP_HEAD_DIMS 64 72 80 96 128 192)

# Paged prefill (FmhaPrefillRunner) and non-paged prefill (FmhaPrefillNpRunner).
set(FMHA_PREFILL_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/flash_attention_v2/xe_fmha_fwd_prefill_kernel.cpp.in")

set(FMHA_PREFILL_NOPAGE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/flash_attention_v2/xe_fmha_fwd_prefill_nopage_kernel.cpp.in")

# FP8 KV-cache prefill path is split into a dedicated runner TU
# (FmhaPrefillFp8Runner) so its heavy e4m3/e5m2 kernel instantiations do not
# inflate the bf16/fp16 prefill TU's peak compiler memory.
set(FMHA_PREFILL_FP8_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/flash_attention_v2/xe_fmha_fwd_prefill_fp8_kernel.cpp.in")

# Per-HEAD_DIM tile shape parameters (TILED_Q, TILED_KV, NUM_SG)
set(FMHA_PREFILL_TILED_Q_64 128)
set(FMHA_PREFILL_TILED_KV_64 64)
set(FMHA_PREFILL_NUM_SG_64 8)

set(FMHA_PREFILL_TILED_Q_96 128)
set(FMHA_PREFILL_TILED_KV_96 64)
set(FMHA_PREFILL_NUM_SG_96 8)

set(FMHA_PREFILL_TILED_Q_128 256)
set(FMHA_PREFILL_TILED_KV_128 32)
set(FMHA_PREFILL_NUM_SG_128 16)

set(FMHA_PREFILL_TILED_Q_192 256)
set(FMHA_PREFILL_TILED_KV_192 64)
set(FMHA_PREFILL_NUM_SG_192 32)

set(FMHA_PREFILL_TILED_Q_256 256)
set(FMHA_PREFILL_TILED_KV_256 64)
set(FMHA_PREFILL_NUM_SG_256 32)

set(FMHA_PREFILL_TILED_Q_512 256)
set(FMHA_PREFILL_TILED_KV_512 64)
set(FMHA_PREFILL_NUM_SG_512 32)
set(FMHA_PREFILL_TILED_OUT_512 256)

# Per-HEAD_DIM tile shape parameters for the NON-PAGED (contiguous ragged) KV
# path (TILED_Q_NP, TILED_KV_NP, NUM_SG_NP). These are kept as a separate set so
# the non-paged path can be tuned independently of the paged path. They are
# initialized to the paged values and can be adjusted later.
set(FMHA_PREFILL_TILED_Q_NP_64 128)
set(FMHA_PREFILL_TILED_KV_NP_64 64)
set(FMHA_PREFILL_NUM_SG_NP_64 8)

set(FMHA_PREFILL_TILED_Q_NP_72 256)
set(FMHA_PREFILL_TILED_KV_NP_72 64)
set(FMHA_PREFILL_NUM_SG_NP_72 16)

set(FMHA_PREFILL_TILED_Q_NP_80 256)
set(FMHA_PREFILL_TILED_KV_NP_80 64)
set(FMHA_PREFILL_NUM_SG_NP_80 16)

set(FMHA_PREFILL_TILED_Q_NP_96 256)
set(FMHA_PREFILL_TILED_KV_NP_96 64)
set(FMHA_PREFILL_NUM_SG_NP_96 16)

set(FMHA_PREFILL_TILED_Q_NP_128 256)
set(FMHA_PREFILL_TILED_KV_NP_128 32)
set(FMHA_PREFILL_NUM_SG_NP_128 16)

set(FMHA_PREFILL_TILED_Q_NP_192 256)
set(FMHA_PREFILL_TILED_KV_NP_192 32)
set(FMHA_PREFILL_NUM_SG_NP_192 16)

# --- Paged prefill + FP8: paged head dims only, bf16 query only. ---
# prefill_paged (16-bit KV) and prefill_fp8 (e4m3/e5m2 KV) are independent
# shared libraries per HEAD_DIM.
foreach(HEAD_DIM ${FMHA_PREFILL_PAGED_HEAD_DIMS})
    set(TILED_Q ${FMHA_PREFILL_TILED_Q_${HEAD_DIM}})
    set(TILED_KV ${FMHA_PREFILL_TILED_KV_${HEAD_DIM}})
    set(NUM_SG ${FMHA_PREFILL_NUM_SG_${HEAD_DIM}})
    if(NOT TILED_Q OR NOT TILED_KV OR NOT NUM_SG)
        message(FATAL_ERROR "Missing paged tile params for prefill HEAD_DIM=${HEAD_DIM}")
    endif()

    # Output-tile head extent. Defaults to HEAD_DIM rounded up to a multiple of
    # 32; a head dim may override it (e.g. 512 uses 256 to chunk the head).
    math(EXPR TILED_OUT "((${HEAD_DIM} + 31) / 32) * 32")
    if(DEFINED FMHA_PREFILL_TILED_OUT_${HEAD_DIM})
        set(TILED_OUT ${FMHA_PREFILL_TILED_OUT_${HEAD_DIM}})
    endif()

    set(GENERATED_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_prefill_paged_kernel_${HEAD_DIM}.cpp")
    configure_file(${FMHA_PREFILL_TEMPLATE} ${GENERATED_FILE} @ONLY)
    list(APPEND device_cpp_common ${GENERATED_FILE})

    set(GENERATED_FP8_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_prefill_fp8_kernel_${HEAD_DIM}.cpp")
    configure_file(${FMHA_PREFILL_FP8_TEMPLATE} ${GENERATED_FP8_FILE} @ONLY)
    list(APPEND device_cpp_common ${GENERATED_FP8_FILE})
endforeach()

# --- Non-paged (no_page) prefill: np head dims only, no fp8, bf16 query only. ---
foreach(HEAD_DIM ${FMHA_PREFILL_NP_HEAD_DIMS})
    set(TILED_Q_NP ${FMHA_PREFILL_TILED_Q_NP_${HEAD_DIM}})
    set(TILED_KV_NP ${FMHA_PREFILL_TILED_KV_NP_${HEAD_DIM}})
    set(NUM_SG_NP ${FMHA_PREFILL_NUM_SG_NP_${HEAD_DIM}})
    if(NOT TILED_Q_NP OR NOT TILED_KV_NP OR NOT NUM_SG_NP)
        message(FATAL_ERROR "Missing non-paged tile params for prefill HEAD_DIM=${HEAD_DIM}")
    endif()

    set(GENERATED_NP_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_prefill_nopage_kernel_${HEAD_DIM}.cpp")
    configure_file(${FMHA_PREFILL_NOPAGE_TEMPLATE} ${GENERATED_NP_FILE} @ONLY)
    list(APPEND device_cpp_common ${GENERATED_NP_FILE})
endforeach()
