# Generate fused gate/up LoRA-B forward kernel instantiation files.
# Each (ELEM_TAG, TILE_TAG) combination is compiled as a separate translation
# unit so the heavy CUTLASS template instantiation parallelizes across the
# build, matching the convention used by the other Xe20 grouped-GEMM kernels
# (see QKVLoraBFwdXe20.cmake).
#
# To add a new tile, register both a tag name in GATE_UP_LORA_B_FWD_TILE_TAGS and
# the matching C++ option-tag type in GATE_UP_LORA_B_FWD_TILE_TYPES here, define
# that option tag in gate_up_lora_b_fwd_types.hpp, and extend the dispatch in
# gate_up_lora_b_fwd_dispatch.hpp / GateUpLoraBFwd.cpp.

set(GATE_UP_LORA_B_FWD_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/gate_up_lora_b_fwd_kernel.cpp.in")
set(GATE_UP_LORA_B_FWD_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/gate_up_lora_b_fwd")
set(GATE_UP_LORA_B_FWD_INST_SRCS)
file(MAKE_DIRECTORY ${GATE_UP_LORA_B_FWD_GEN_DIR})

# Data-type axis (fp16 / bf16 only -- no fp32 path).
set(GATE_UP_LORA_B_FWD_ELEM_TAGS half bf16)
set(GATE_UP_LORA_B_FWD_ELEM_TORCH_TYPES "at::Half" "at::BFloat16")

# Tile-configuration axis. Each tag maps to an option-tag type defined in
# gate_up_lora_b_fwd_types.hpp.
set(GATE_UP_LORA_B_FWD_TILE_TAGS tall)
set(GATE_UP_LORA_B_FWD_TILE_TYPES "gate_up_lora_b_fwd_impl::GateUpLoraBFwdTileTall")

list(LENGTH GATE_UP_LORA_B_FWD_ELEM_TAGS _num_elems)
math(EXPR _num_elems "${_num_elems} - 1")
list(LENGTH GATE_UP_LORA_B_FWD_TILE_TAGS _num_tiles)
math(EXPR _num_tiles "${_num_tiles} - 1")

foreach(_ei RANGE ${_num_elems})
    list(GET GATE_UP_LORA_B_FWD_ELEM_TAGS ${_ei} ELEM_TAG)
    list(GET GATE_UP_LORA_B_FWD_ELEM_TORCH_TYPES ${_ei} ELEM_TORCH_TYPE)

    foreach(_ti RANGE ${_num_tiles})
        list(GET GATE_UP_LORA_B_FWD_TILE_TAGS ${_ti} TILE_TAG)
        list(GET GATE_UP_LORA_B_FWD_TILE_TYPES ${_ti} TILE_TYPE)

        set(GEN_SRC "${GATE_UP_LORA_B_FWD_GEN_DIR}/gate_up_lora_b_fwd_kernel_${ELEM_TAG}_${TILE_TAG}.cpp")
        configure_file(${GATE_UP_LORA_B_FWD_TEMPLATE} ${GEN_SRC} @ONLY)
        list(APPEND GATE_UP_LORA_B_FWD_INST_SRCS ${GEN_SRC})
    endforeach()
endforeach()

list(APPEND ATen_XPU_SYCL_XE20 ${GATE_UP_LORA_B_FWD_INST_SRCS})
