# Generate SGEMM LoRA-B forward kernel instantiation files.
# Each (ELEM_TAG, TILE_TAG) combination is compiled as a separate translation
# unit so the heavy CUTLASS template instantiation parallelizes across the
# build, matching the convention used by the other Xe20 grouped-GEMM kernels
# (see GroupGemmXe20.cmake / SGEMMLoraAFwdXe20.cmake).
#
# To add a new tile, register both a tag name in SGEMM_LORA_B_FWD_TILE_TAGS and
# the matching C++ option-tag type in SGEMM_LORA_B_FWD_TILE_TYPES here, define
# that option tag in sgemm_lora_b_fwd_types.hpp, and extend the dispatch in
# sgemm_lora_b_fwd_dispatch.hpp / SGEMMLoraBFwd.cpp.

set(SGEMM_LORA_B_FWD_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/sgemm_lora_b_fwd_kernel.cpp.in")
set(SGEMM_LORA_B_FWD_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/sgemm_lora_b_fwd")
set(SGEMM_LORA_B_FWD_INST_SRCS)
file(MAKE_DIRECTORY ${SGEMM_LORA_B_FWD_GEN_DIR})

# Data-type axis (fp16 / bf16 only -- no fp32 path).
set(SGEMM_LORA_B_FWD_ELEM_TAGS half bf16)
set(SGEMM_LORA_B_FWD_ELEM_TORCH_TYPES "at::Half" "at::BFloat16")

# Tile-configuration axis. Each tag maps to an option-tag type defined in
# sgemm_lora_b_fwd_types.hpp.
set(SGEMM_LORA_B_FWD_TILE_TAGS large)
set(SGEMM_LORA_B_FWD_TILE_TYPES "sgemm_lora_b_fwd_impl::LoraBFwdTileLarge")

list(LENGTH SGEMM_LORA_B_FWD_ELEM_TAGS _num_elems)
math(EXPR _num_elems "${_num_elems} - 1")
list(LENGTH SGEMM_LORA_B_FWD_TILE_TAGS _num_tiles)
math(EXPR _num_tiles "${_num_tiles} - 1")

foreach(_ei RANGE ${_num_elems})
    list(GET SGEMM_LORA_B_FWD_ELEM_TAGS ${_ei} ELEM_TAG)
    list(GET SGEMM_LORA_B_FWD_ELEM_TORCH_TYPES ${_ei} ELEM_TORCH_TYPE)

    foreach(_ti RANGE ${_num_tiles})
        list(GET SGEMM_LORA_B_FWD_TILE_TAGS ${_ti} TILE_TAG)
        list(GET SGEMM_LORA_B_FWD_TILE_TYPES ${_ti} TILE_TYPE)

        set(GEN_SRC "${SGEMM_LORA_B_FWD_GEN_DIR}/sgemm_lora_b_fwd_kernel_${ELEM_TAG}_${TILE_TAG}.cpp")
        configure_file(${SGEMM_LORA_B_FWD_TEMPLATE} ${GEN_SRC} @ONLY)
        list(APPEND SGEMM_LORA_B_FWD_INST_SRCS ${GEN_SRC})
    endforeach()
endforeach()

list(APPEND ATen_XPU_SYCL_XE20 ${SGEMM_LORA_B_FWD_INST_SRCS})
