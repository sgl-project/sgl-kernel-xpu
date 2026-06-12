# Generate FMHA prefill kernel instantiation files.
# Each HEAD_DIM is compiled as a separate translation unit to parallelize
# and speed up compilation.
#
# The paged and non-paged (contiguous ragged) KV paths support INDEPENDENT sets
# of head dimensions. A single FmhaPrefillRunner<HD> translation unit contains
# both branches, each gated at compile time by @EMIT_PAGED@ / @EMIT_NP@ so a head
# dim only emits the branch(es) it actually supports. The cmake below iterates
# the union of both lists and emits per-head-dim EMIT flags into the template.
set(FMHA_PREFILL_PAGED_HEAD_DIMS 64 96 128 192 256 512)
set(FMHA_PREFILL_NP_HEAD_DIMS 64 72 96 128)

set(FMHA_PREFILL_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/xe_fmha_fwd_prefill_kernel.cpp.in")

# Per-HEAD_DIM tile shape parameters (TILED_Q, TILED_KV, NUM_SG)
set(FMHA_PREFILL_TILED_Q_64 128)
set(FMHA_PREFILL_TILED_KV_64 64)
set(FMHA_PREFILL_NUM_SG_64 8)

set(FMHA_PREFILL_TILED_Q_96 128)
set(FMHA_PREFILL_TILED_KV_96 64)
set(FMHA_PREFILL_NUM_SG_96 8)

set(FMHA_PREFILL_TILED_Q_128 128)
set(FMHA_PREFILL_TILED_KV_128 64)
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

set(FMHA_PREFILL_TILED_Q_NP_96 256)
set(FMHA_PREFILL_TILED_KV_NP_96 64)
set(FMHA_PREFILL_NUM_SG_NP_96 16)

set(FMHA_PREFILL_TILED_Q_NP_128 256)
set(FMHA_PREFILL_TILED_KV_NP_128 32)
set(FMHA_PREFILL_NUM_SG_NP_128 16)

# Compile the union of the paged and non-paged head-dim sets. Each generated TU
# is told via EMIT_PAGED / EMIT_NP which branch(es) to compile; the disabled
# branch is preprocessed out, so its tile params are never used (placeholders
# below just keep the @VAR@ substitutions non-empty for cleanliness).
set(FMHA_PREFILL_HEAD_DIMS ${FMHA_PREFILL_PAGED_HEAD_DIMS} ${FMHA_PREFILL_NP_HEAD_DIMS})
list(REMOVE_DUPLICATES FMHA_PREFILL_HEAD_DIMS)
list(SORT FMHA_PREFILL_HEAD_DIMS COMPARE NATURAL)

foreach(HEAD_DIM ${FMHA_PREFILL_HEAD_DIMS})
    if(HEAD_DIM IN_LIST FMHA_PREFILL_PAGED_HEAD_DIMS)
        set(EMIT_PAGED 1)
        set(TILED_Q ${FMHA_PREFILL_TILED_Q_${HEAD_DIM}})
        set(TILED_KV ${FMHA_PREFILL_TILED_KV_${HEAD_DIM}})
        set(NUM_SG ${FMHA_PREFILL_NUM_SG_${HEAD_DIM}})
        if(NOT TILED_Q OR NOT TILED_KV OR NOT NUM_SG)
            message(FATAL_ERROR "Missing paged tile params for prefill HEAD_DIM=${HEAD_DIM}")
        endif()
    else()
        set(EMIT_PAGED 0)
        set(TILED_Q 128)
        set(TILED_KV 64)
        set(NUM_SG 8)
    endif()

    if(HEAD_DIM IN_LIST FMHA_PREFILL_NP_HEAD_DIMS)
        set(EMIT_NP 1)
        set(TILED_Q_NP ${FMHA_PREFILL_TILED_Q_NP_${HEAD_DIM}})
        set(TILED_KV_NP ${FMHA_PREFILL_TILED_KV_NP_${HEAD_DIM}})
        set(NUM_SG_NP ${FMHA_PREFILL_NUM_SG_NP_${HEAD_DIM}})
        if(NOT TILED_Q_NP OR NOT TILED_KV_NP OR NOT NUM_SG_NP)
            message(FATAL_ERROR "Missing non-paged tile params for prefill HEAD_DIM=${HEAD_DIM}")
        endif()
    else()
        set(EMIT_NP 0)
        set(TILED_Q_NP 128)
        set(TILED_KV_NP 64)
        set(NUM_SG_NP 8)
    endif()

    set(GENERATED_FILE
        "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_prefill_kernel_${HEAD_DIM}.cpp")
    configure_file(${FMHA_PREFILL_TEMPLATE} ${GENERATED_FILE} @ONLY)
    list(APPEND device_cpp_common ${GENERATED_FILE})
endforeach()
