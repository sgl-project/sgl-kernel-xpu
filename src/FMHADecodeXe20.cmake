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

# The non-paged decode KV-tile size now lives in the shared header
# sycl/kernels/flash_attention_v2/fmha_tile_dispatch.h (decode_tiled_kv_np),
# consumed both by the AOT template (compile-time lookup by HEAD_DIM) and the
# runtime-JIT wrapper.

# Paged decode (FmhaDecodeRunner) and non-paged decode (FmhaDecodeNpRunner).
set(FMHA_DECODE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/flash_attention_v2/xe_fmha_fwd_decode_kernel.cpp.in")

set(FMHA_DECODE_NOPAGE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/flash_attention_v2/xe_fmha_fwd_decode_nopage_kernel.cpp.in")

set(FMHA_SPLIT_DECODE_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/flash_attention_v2/xe_fmha_fwd_split_decode_kernel.cpp.in")

# FP8 KV-cache paths are split into dedicated runner TUs (FmhaDecodeFp8Runner /
# FmhaSplitDecodeFp8Runner) so their heavy e4m3/e5m2 kernel instantiations do not
# inflate the 16-bit-KV decode / split-decode TUs' peak compiler memory. The KV
# cache is fp8 while the query dtype is bf16 or fp16 (each its own TU / .so).
set(FMHA_DECODE_FP8_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/flash_attention_v2/xe_fmha_fwd_decode_fp8_kernel.cpp.in")

set(FMHA_SPLIT_DECODE_FP8_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/flash_attention_v2/xe_fmha_fwd_split_decode_fp8_kernel.cpp.in")

# woq mxfp4 KV-cache decode path (packed E2M1 + E8M0 block scale, bf16 query
# only). Non-split only; the split path forces non-split for mxfp4.
set(FMHA_DECODE_MXFP4_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/kernels/flash_attention_v2/xe_fmha_fwd_decode_mxfp4_kernel.cpp.in")

# 16-bit query element tags. Each tag produces INDEPENDENT shared libraries for
# the decode / split-decode / non-paged paths so bf16 and fp16 do not share a
# translation unit. ELEM_TYPE selects the cutlass query/KV/out element type.
#
# Generated file name order (so name = lib<file>.so):
#   xe_fmha_fwd_<op>_<page|nopage>_<tileconfig>_<Qtype>_<KVtype>
# where op in {decode, split_decode}, tileconfig = QG_HD[_PS], and the trailing
# tags are the query dtype then the KV-cache dtype (16-bit: KV==Q; fp8: KV=fp8).
set(FMHA_DECODE_ELEM_TAGS bf16 fp16)

foreach(QG_SZ ${FMHA_DECODE_QG_SIZES})
    # --- Paged decode + split-decode: paged head dims only. ---
    # Each (QG, HEAD_DIM, PAGE_SIZE) yields independent shared libraries split by
    # KV-cache dtype:
    #   decode_page / split_decode_page   (16-bit KV: *_bf16_bf16 / *_fp16_fp16)
    #   decode_page / split_decode_page   (fp8 KV:    *_bf16_fp8, bf16 query only)
    foreach(HEAD_DIM ${FMHA_DECODE_PAGED_HEAD_DIMS})
        foreach(PAGE_SIZE ${FMHA_DECODE_PAGE_SIZES})
            foreach(ELEM_TAG ${FMHA_DECODE_ELEM_TAGS})
                if(ELEM_TAG STREQUAL "bf16")
                    set(ELEM_TYPE "cutlass::bfloat16_t")
                else()
                    set(ELEM_TYPE "cutlass::half_t")
                endif()
                # Trailing <Qtype>_<KVtype> tags. 16-bit KV mirrors the query dtype;
                # the fp8 path keeps the query dtype and sets KV to fp8.
                set(DT16 "${ELEM_TAG}_${ELEM_TAG}")
                set(DTFP8 "${ELEM_TAG}_fp8")

                set(GENERATED_FILE
                    "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_decode_page_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}_${DT16}.cpp")
                configure_file(${FMHA_DECODE_TEMPLATE} ${GENERATED_FILE} @ONLY)
                list(APPEND device_cpp_xe20 ${GENERATED_FILE})

                set(GENERATED_SPLIT_FILE
                    "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_split_decode_page_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}_${DT16}.cpp")
                configure_file(${FMHA_SPLIT_DECODE_TEMPLATE} ${GENERATED_SPLIT_FILE} @ONLY)
                list(APPEND device_cpp_xe20 ${GENERATED_SPLIT_FILE})

                # FP8 KV cache: bf16 query only (KV dtype = fp8, Q dtype = bf16).
                # fp16 query + fp8 KV is intentionally not built.
                if(ELEM_TAG STREQUAL "bf16")
                    set(GENERATED_FP8_FILE
                        "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_decode_page_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}_${DTFP8}.cpp")
                    configure_file(${FMHA_DECODE_FP8_TEMPLATE} ${GENERATED_FP8_FILE} @ONLY)
                    list(APPEND device_cpp_xe20 ${GENERATED_FP8_FILE})

                    set(GENERATED_SPLIT_FP8_FILE
                        "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_split_decode_page_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}_${DTFP8}.cpp")
                    configure_file(${FMHA_SPLIT_DECODE_FP8_TEMPLATE} ${GENERATED_SPLIT_FP8_FILE} @ONLY)
                    list(APPEND device_cpp_xe20 ${GENERATED_SPLIT_FP8_FILE})

                    # woq mxfp4 KV cache: bf16 query only, non-split decode only.
                    set(GENERATED_MXFP4_FILE
                        "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_decode_page_${QG_SZ}_${HEAD_DIM}_${PAGE_SIZE}_${ELEM_TAG}_mxfp4.cpp")
                    configure_file(${FMHA_DECODE_MXFP4_TEMPLATE} ${GENERATED_MXFP4_FILE} @ONLY)
                    list(APPEND device_cpp_xe20 ${GENERATED_MXFP4_FILE})
                endif()
            endforeach()
        endforeach()
    endforeach()

    # --- Non-paged (no_page) decode: np head dims only, no page size, no fp8. ---
    # 16-bit KV only (KV==Q: *_bf16_bf16 / *_fp16_fp16), no split-KV. The KV-tile
    # size comes from the shared header fmha_tile_dispatch.h (decode_tiled_kv_np),
    # looked up in the template by HEAD_DIM at compile time.
    foreach(HEAD_DIM ${FMHA_DECODE_NP_HEAD_DIMS})

        foreach(ELEM_TAG ${FMHA_DECODE_ELEM_TAGS})
            if(ELEM_TAG STREQUAL "bf16")
                set(ELEM_TYPE "cutlass::bfloat16_t")
            else()
                set(ELEM_TYPE "cutlass::half_t")
            endif()
            set(DT16 "${ELEM_TAG}_${ELEM_TAG}")

            set(GENERATED_NP_FILE
                "${CMAKE_CURRENT_BINARY_DIR}/sycl/xe_fmha_fwd_decode_nopage_${QG_SZ}_${HEAD_DIM}_${DT16}.cpp")
            configure_file(${FMHA_DECODE_NOPAGE_TEMPLATE} ${GENERATED_NP_FILE} @ONLY)
            list(APPEND device_cpp_xe20 ${GENERATED_NP_FILE})
        endforeach()
    endforeach()
endforeach()
