# Generate SGEMM LoRA-A forward kernel instantiation files.
# Each ELEM_TAG is compiled as a separate library to parallelize and speed up
# compilation, matching the convention used by the other Xe20 grouped-GEMM
# kernels (see GroupGemmXe20.cmake).

set(SGEMM_LORA_A_FWD_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/sycl/sgemm_lora_a_fwd_kernel.cpp.in")
set(SGEMM_LORA_A_FWD_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/sgemm_lora_a_fwd")
set(SGEMM_LORA_A_FWD_INST_SRCS)
file(MAKE_DIRECTORY ${SGEMM_LORA_A_FWD_GEN_DIR})

set(SGEMM_LORA_A_FWD_ELEM_TAGS half bf16)
set(SGEMM_LORA_A_FWD_ELEM_CUTLASS_TYPES "at::Half" "at::BFloat16")

list(LENGTH SGEMM_LORA_A_FWD_ELEM_TAGS _num_elems)
math(EXPR _num_elems "${_num_elems} - 1")

foreach(_idx RANGE ${_num_elems})
    list(GET SGEMM_LORA_A_FWD_ELEM_TAGS ${_idx} ELEM_TAG)
    list(GET SGEMM_LORA_A_FWD_ELEM_CUTLASS_TYPES ${_idx} ELEM_CUTLASS_TYPE)

    set(GEN_SRC "${SGEMM_LORA_A_FWD_GEN_DIR}/sgemm_lora_a_fwd_kernel_${ELEM_TAG}.cpp")
    configure_file(${SGEMM_LORA_A_FWD_TEMPLATE} ${GEN_SRC} @ONLY)
    list(APPEND SGEMM_LORA_A_FWD_INST_SRCS ${GEN_SRC})
endforeach()

list(APPEND ATen_XPU_SYCL_XE20 ${SGEMM_LORA_A_FWD_INST_SRCS})
