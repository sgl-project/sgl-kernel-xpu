# Generate dtype instantiations independently to bound compiler memory.
set(INKLING_GATE_PROJECTION_TEMPLATE
    "${CMAKE_CURRENT_SOURCE_DIR}/sycl/inkling_moe_gate_projection_kernel_Xe35.cpp.in")
set(INKLING_GATE_PROJECTION_GEN_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/generated/inkling_moe_gate_projection")
file(MAKE_DIRECTORY ${INKLING_GATE_PROJECTION_GEN_DIR})

set(INKLING_GATE_ELEM_TAGS bf16 fp16)
set(INKLING_GATE_ELEMENT_TYPES cutlass::bfloat16_t cutlass::half_t)
list(LENGTH INKLING_GATE_ELEM_TAGS _inkling_gate_elem_count)
math(EXPR _inkling_gate_elem_last "${_inkling_gate_elem_count} - 1")
foreach(_elem_index RANGE ${_inkling_gate_elem_last})
  list(GET INKLING_GATE_ELEM_TAGS ${_elem_index} ELEM_TAG)
  list(GET INKLING_GATE_ELEMENT_TYPES ${_elem_index} ELEMENT_TYPE)
  set(GEN_SRC
      "${INKLING_GATE_PROJECTION_GEN_DIR}/inkling_moe_gate_projection_${ELEM_TAG}Xe35.cpp")
  configure_file(${INKLING_GATE_PROJECTION_TEMPLATE} ${GEN_SRC} @ONLY)
  # Keep generated providers separate until CMakeLists has appended the
  # hand-written dispatcher. BuildOnLinux relies on provider libraries
  # appearing after their consumers so --as-needed records them in DT_NEEDED.
  list(APPEND INKLING_GATE_PROJECTION_SRCS ${GEN_SRC})
endforeach()
