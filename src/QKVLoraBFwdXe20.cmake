# Generate fused QKV LoRA-B forward kernel instantiation files.
# Each (ELEM_TAG, TILE_TAG) combination is compiled as a separate translation
# unit so the heavy CUTLASS template instantiation parallelizes across the
# build, matching the convention used by the other Xe20 grouped-GEMM kernels
# (see SGEMMLoraBFwdXe20.cmake).
#
# To add a new tile, register both a tag name in QKV_LORA_B_FWD_TILE_TAGS and
# the matching C++ option-tag type in QKV_LORA_B_FWD_TILE_TYPES here, define
# that option tag in qkv_lora_b_fwd_types.hpp, and extend the dispatch in
# qkv_lora_b_fwd_dispatch.hpp / QKVLoraBFwd.cpp.

set(QKV_LORA_B_FWD_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/qkv_lora_b_fwd_kernel.cpp.in")
set(QKV_LORA_B_FWD_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/qkv_lora_b_fwd")
set(QKV_LORA_B_FWD_INST_SRCS)
file(MAKE_DIRECTORY ${QKV_LORA_B_FWD_GEN_DIR})

# Data-type axis (fp16 / bf16 only -- no fp32 path).
set(QKV_LORA_B_FWD_ELEM_TAGS half bf16)
set(QKV_LORA_B_FWD_ELEM_TORCH_TYPES "at::Half" "at::BFloat16")

# Tile-configuration axis. Each tag maps to an option-tag type defined in
# qkv_lora_b_fwd_types.hpp.
set(QKV_LORA_B_FWD_TILE_TAGS tall)
set(QKV_LORA_B_FWD_TILE_TYPES "qkv_lora_b_fwd_impl::QKVLoraBFwdTileTall")

list(LENGTH QKV_LORA_B_FWD_ELEM_TAGS _num_elems)
math(EXPR _num_elems "${_num_elems} - 1")
list(LENGTH QKV_LORA_B_FWD_TILE_TAGS _num_tiles)
math(EXPR _num_tiles "${_num_tiles} - 1")

foreach(_ei RANGE ${_num_elems})
    list(GET QKV_LORA_B_FWD_ELEM_TAGS ${_ei} ELEM_TAG)
    list(GET QKV_LORA_B_FWD_ELEM_TORCH_TYPES ${_ei} ELEM_TORCH_TYPE)

    foreach(_ti RANGE ${_num_tiles})
        list(GET QKV_LORA_B_FWD_TILE_TAGS ${_ti} TILE_TAG)
        list(GET QKV_LORA_B_FWD_TILE_TYPES ${_ti} TILE_TYPE)

        set(GEN_SRC "${QKV_LORA_B_FWD_GEN_DIR}/qkv_lora_b_fwd_kernel_${ELEM_TAG}_${TILE_TAG}.cpp")
        configure_file(${QKV_LORA_B_FWD_TEMPLATE} ${GEN_SRC} @ONLY)
        list(APPEND QKV_LORA_B_FWD_INST_SRCS ${GEN_SRC})
    endforeach()
endforeach()

list(APPEND ATen_XPU_SYCL_XE20 ${QKV_LORA_B_FWD_INST_SRCS})
