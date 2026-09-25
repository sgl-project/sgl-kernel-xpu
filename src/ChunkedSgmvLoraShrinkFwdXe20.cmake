# Generate chunked-SGMV LoRA shrink grouped-GEMM kernel instantiation files.
# Each (ELEM_TAG, TILE_TAG) combination is compiled as a separate translation
# unit so the heavy CUTLASS template instantiation parallelizes across the
# build, matching the convention used by the other Xe20 grouped-GEMM kernels
# (see SGEMMLoraAFwdXe20.cmake).
#
# This is a dedicated small-N tile variant of the LoRA-A "shrink" GEMM; the
# merged sgemm_lora_a_fwd kernel is left untouched.
#
# To add a new tile, register both a tag name in
# CHUNKED_SGMV_LORA_SHRINK_FWD_TILE_TAGS and the matching C++ option-tag type in
# CHUNKED_SGMV_LORA_SHRINK_FWD_TILE_TYPES here, define that option tag in
# chunked_sgmv_lora_shrink_fwd_types.hpp, and extend the dispatch in
# chunked_sgmv_lora_shrink_fwd_dispatch.hpp / ChunkedSgmvLoraShrinkFwd.cpp.

set(CHUNKED_SGMV_LORA_SHRINK_FWD_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/chunked_sgmv_lora_shrink_fwd_kernel.cpp.in")
set(CHUNKED_SGMV_LORA_SHRINK_FWD_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/chunked_sgmv_lora_shrink_fwd")
set(CHUNKED_SGMV_LORA_SHRINK_FWD_INST_SRCS)
file(MAKE_DIRECTORY ${CHUNKED_SGMV_LORA_SHRINK_FWD_GEN_DIR})

# Data-type axis (fp16 / bf16 only -- no fp32 path).
set(CHUNKED_SGMV_LORA_SHRINK_FWD_ELEM_TAGS half bf16)
set(CHUNKED_SGMV_LORA_SHRINK_FWD_ELEM_TORCH_TYPES "at::Half" "at::BFloat16")

# Tile-configuration axis. Each tag maps to an option-tag type defined in
# chunked_sgmv_lora_shrink_fwd_types.hpp.
set(CHUNKED_SGMV_LORA_SHRINK_FWD_TILE_TAGS small)
set(CHUNKED_SGMV_LORA_SHRINK_FWD_TILE_TYPES "chunked_sgmv_lora_shrink_fwd_impl::ChunkedShrinkTileSmall")

list(LENGTH CHUNKED_SGMV_LORA_SHRINK_FWD_ELEM_TAGS _num_elems)
math(EXPR _num_elems "${_num_elems} - 1")
list(LENGTH CHUNKED_SGMV_LORA_SHRINK_FWD_TILE_TAGS _num_tiles)
math(EXPR _num_tiles "${_num_tiles} - 1")

foreach(_ei RANGE ${_num_elems})
    list(GET CHUNKED_SGMV_LORA_SHRINK_FWD_ELEM_TAGS ${_ei} ELEM_TAG)
    list(GET CHUNKED_SGMV_LORA_SHRINK_FWD_ELEM_TORCH_TYPES ${_ei} ELEM_TORCH_TYPE)

    foreach(_ti RANGE ${_num_tiles})
        list(GET CHUNKED_SGMV_LORA_SHRINK_FWD_TILE_TAGS ${_ti} TILE_TAG)
        list(GET CHUNKED_SGMV_LORA_SHRINK_FWD_TILE_TYPES ${_ti} TILE_TYPE)

        set(GEN_SRC "${CHUNKED_SGMV_LORA_SHRINK_FWD_GEN_DIR}/chunked_sgmv_lora_shrink_fwd_kernel_${ELEM_TAG}_${TILE_TAG}.cpp")
        configure_file(${CHUNKED_SGMV_LORA_SHRINK_FWD_TEMPLATE} ${GEN_SRC} @ONLY)
        list(APPEND CHUNKED_SGMV_LORA_SHRINK_FWD_INST_SRCS ${GEN_SRC})
    endforeach()
endforeach()

list(APPEND ATen_XPU_SYCL_XE20 ${CHUNKED_SGMV_LORA_SHRINK_FWD_INST_SRCS})
list(APPEND ATen_XPU_SYCL_XE35 ${CHUNKED_SGMV_LORA_SHRINK_FWD_INST_SRCS})
