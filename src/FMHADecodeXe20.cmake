# Generate FMHA decode kernel instantiation files.
# Each (QG_SZ, HEAD_DIM, PAGE_SIZE) combination is compiled as a separate
# library to parallelize and speed up compilation.
#
# The paged and non-paged (contiguous ragged) KV paths support INDEPENDENT sets
# of head dimensions. A single FmhaDecodeRunner<QG,HD,PS> translation unit
# contains both branches, each gated at compile time by @EMIT_PAGED@ / @EMIT_NP@
# so a head dim only emits the branch(es) it actually supports. The cmake below
# iterates the union of both lists and emits per-head-dim EMIT flags into the
# template.
set(FMHA_DECODE_QG_SIZES 1 2 4 8 16)
set(FMHA_DECODE_PAGED_HEAD_DIMS 64 96 128 192 256 512)
set(FMHA_DECODE_NP_HEAD_DIMS 64 72 96 128 192)
set(FMHA_DECODE_PAGE_SIZES 64 128)

# Per-HEAD_DIM KV-tile size for the NON-PAGED (contiguous ragged) decode path.
# The paged decode kernel uses PAGE_SIZE as its KV tile; the non-paged path has
# no natural page size, so it gets its own KV-tile constant that can be tuned
# independently. The non-paged kernel is only emitted in the PAGE_SIZE==128
# translation unit (see xe_fmha_fwd_decode_kernel.cpp.in) and the host routes
# the non-paged decode launch to that TU. Must be a multiple of 16. Only the
# head dims in FMHA_DECODE_NP_HEAD_DIMS need an entry here.
# Note: Larger head dimensions require smaller KV tiles to avoid running out of
# registers/local memory on Level Zero backend (UR_RESULT_ERROR_OUT_OF_RESOURCES).
set(FMHA_DECODE_TILED_KV_NP_64 512)
set(FMHA_DECODE_TILED_KV_NP_72 512)
set(FMHA_DECODE_TILED_KV_NP_96 512)
set(FMHA_DECODE_TILED_KV_NP_128 512)
set(FMHA_DECODE_TILED_KV_NP_192 512)

set(FMHA_DECODE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/xe_fmha_fwd_decode_kernel.cpp.in")

set(FMHA_SPLIT_DECODE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/xe_fmha_fwd_split_decode_kernel.cpp.in")

# FP8 KV-cache paths are split into dedicated runner TUs (FmhaDecodeFp8Runner /
# FmhaSplitDecodeFp8Runner) so their heavy e4m3/e5m2 kernel instantiations do not
# inflate the bf16/fp16 decode / split-decode TUs' peak compiler memory.
set(FMHA_DECODE_FP8_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/xe_fmha_fwd_decode_fp8_kernel.cpp.in")

set(FMHA_SPLIT_DECODE_FP8_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/xe_fmha_fwd_split_decode_fp8_kernel.cpp.in")

# Compile the union of the paged and non-paged head-dim sets. Each generated TU
# is told via EMIT_PAGED / EMIT_NP which branch(es) to compile; the disabled
# branch is preprocessed out. The split-decode kernel is a paged-only (split-KV)
# optimization that takes no non-paged branch, but it is emitted for the full
# union because the shared DISPATCH_DECODE_KERNEL macro references
# FmhaSplitDecodeRunner<QG,HD,PS> for every dispatched head dim (the symbol must
# exist at link time even though the non-paged head dims never call it).
set(FMHA_DECODE_HEAD_DIMS ${FMHA_DECODE_PAGED_HEAD_DIMS} ${FMHA_DECODE_NP_HEAD_DIMS})
list(REMOVE_DUPLICATES FMHA_DECODE_HEAD_DIMS)
list(SORT FMHA_DECODE_HEAD_DIMS COMPARE NATURAL)

# FP8 KV cache decode/split-decode kernels are instantiated for both bf16 and
# fp16 queries, which roughly doubles the number of fp8 translation units to
# compile. This option lets a build drop the fp16-query instantiation to cut
# compile time; it is OFF by default so an fp16 query against an fp8 KV cache
# raises a runtime TORCH_CHECK unless explicitly enabled. Emitted into the
# templates as @EMIT_FP8_FP16@.
option(SGL_FMHA_FP8_KV_ENABLE_FP16 "Compile fp16-query fp8 KV cache kernels" OFF)
if(SGL_FMHA_FP8_KV_ENABLE_FP16)
    set(EMIT_FP8_FP16 1)
else()
    set(EMIT_FP8_FP16 0)
endif()

foreach(QG_SZ ${FMHA_DECODE_QG_SIZES})
    foreach(HEAD_DIM ${FMHA_DECODE_HEAD_DIMS})
        if(HEAD_DIM IN_LIST FMHA_DECODE_PAGED_HEAD_DIMS)
            set(EMIT_PAGED 1)
        else()
            set(EMIT_PAGED 0)
        endif()

        if(HEAD_DIM IN_LIST FMHA_DECODE_NP_HEAD_DIMS)
            set(EMIT_NP 1)
            set(TILED_KV_NP ${FMHA_DECODE_TILED_KV_NP_${HEAD_DIM}})
            if(NOT TILED_KV_NP)
                message(FATAL_ERROR "Missing non-paged KV tile (FMHA_DECODE_TILED_KV_NP_${HEAD_DIM}) for decode HEAD_DIM=${HEAD_DIM}")
            endif()
        else()
            set(EMIT_NP 0)
            set(TILED_KV_NP 16)
        endif()

        foreach(PAGE_SIZE ${FMHA_DECODE_PAGE_SIZES})
            set(GENERATED_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_decode_kernel_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}.cpp")
            configure_file(${FMHA_DECODE_TEMPLATE} ${GENERATED_FILE} @ONLY)
            list(APPEND device_cpp_common ${GENERATED_FILE})

            # FP8 paged decode runner: only the paged head dims forward to it
            # (the fp8 branch lives inside #if @EMIT_PAGED@ in the wrapper).
            if(EMIT_PAGED)
                set(GENERATED_FP8_FILE
                    "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_decode_fp8_kernel_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}.cpp")
                configure_file(${FMHA_DECODE_FP8_TEMPLATE} ${GENERATED_FP8_FILE} @ONLY)
                list(APPEND device_cpp_common ${GENERATED_FP8_FILE})
            endif()

            set(GENERATED_SPLIT_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_split_decode_kernel_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}.cpp")
            configure_file(${FMHA_SPLIT_DECODE_TEMPLATE} ${GENERATED_SPLIT_FILE} @ONLY)
            list(APPEND device_cpp_common ${GENERATED_SPLIT_FILE})

            # FP8 split-decode runner: the split wrapper forwards to it
            # unconditionally, so it is emitted for the full head-dim union to
            # match FmhaSplitDecodeRunner symbol availability.
            set(GENERATED_SPLIT_FP8_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_split_decode_fp8_kernel_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}.cpp")
            configure_file(${FMHA_SPLIT_DECODE_FP8_TEMPLATE} ${GENERATED_SPLIT_FP8_FILE} @ONLY)
            list(APPEND device_cpp_common ${GENERATED_SPLIT_FP8_FILE})
        endforeach()
    endforeach()
endforeach()
