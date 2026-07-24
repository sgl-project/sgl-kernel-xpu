# Build on Linux

set(SGL_OPS_LIBRARIES)
set(SYCL_LINK_LIBRARIES_KEYWORD PRIVATE)

macro(setup_common_libraries)
  Python3_add_library(
    common_ops
    MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI
    ${ATen_XPU_CPP_SRCS})
  install(TARGETS common_ops LIBRARY DESTINATION sgl_kernel)
  set_target_properties(common_ops PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
  list(APPEND SGL_OPS_LIBRARIES common_ops)
endmacro()

setup_common_libraries()

if(SYCL_COMPILER_VERSION GREATER_EQUAL 20250806)
  set(COMMON_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS})
  set(COMMON_DEVICE_LINK_FLAGS ${COMMON_DEVICE_LINK_FLAGS} -Xspirv-translator)
  set(COMMON_DEVICE_LINK_FLAGS
      ${COMMON_DEVICE_LINK_FLAGS}
      -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate)
else()
  message(FATAL_ERROR
      "SYCL compiler version must be >= 20250806, "
      "but got ${SYCL_COMPILER_VERSION}")
endif()

get_filename_component(SGL_SYCL_COMPILER_DIR "${SYCL_EXECUTABLE}" DIRECTORY)
find_program(SGL_SYCL_ESIMD_HOST_COMPILER
  NAMES icpx
  HINTS "${SGL_SYCL_COMPILER_DIR}"
  NO_DEFAULT_PATH)
if(NOT SGL_SYCL_ESIMD_HOST_COMPILER)
  find_program(SGL_SYCL_ESIMD_HOST_COMPILER NAMES icpx)
endif()

# common kernels
foreach(sycl_src ${ATen_XPU_SYCL_COMMON})
  get_filename_component(name ${sycl_src} NAME_WLE REALPATH)
  set(sycl_lib sgl-ops-sycl-${name})
  set(_saved_SYCL_HOST_COMPILER "${SYCL_HOST_COMPILER}")
  if(name STREQUAL "InklingRelativeHelpers")
    if(NOT SGL_SYCL_ESIMD_HOST_COMPILER)
      message(FATAL_ERROR "InklingRelativeHelpers uses ESIMD and requires icpx as the SYCL host compiler")
    endif()
    set(SYCL_HOST_COMPILER "${SGL_SYCL_ESIMD_HOST_COMPILER}")
  endif()
  sycl_add_library(
    ${sycl_lib}
    ${SYCL_OFFLINE_COMPILER_FLAGS}
    ${COMMON_DEVICE_LINK_FLAGS}
    SHARED
    SYCL_SOURCES ${sycl_src})
  set(SYCL_HOST_COMPILER "${_saved_SYCL_HOST_COMPILER}")
  # Inkling kernels register through scoped extensions, so common_ops does not
  # need to load their SYCL libraries transitively.
  if(NOT name STREQUAL "InklingSconv"
      AND NOT name STREQUAL "InklingAttnPrologue"
      AND NOT name STREQUAL "InklingMoEGate"
      AND NOT name STREQUAL "InklingRelativeAttention")
    target_link_libraries(common_ops PUBLIC ${sycl_lib})
  endif()
  list(APPEND SGL_OPS_LIBRARIES ${sycl_lib})

  # Decouple with PyTorch cmake definition.
  set(sycl_install_args LIBRARY DESTINATION sgl_kernel)
  if(name STREQUAL "InklingSconv")
    list(APPEND sycl_install_args COMPONENT inkling_sconv)
  elseif(name STREQUAL "InklingAttnPrologue")
    list(APPEND sycl_install_args COMPONENT inkling_attn_prologue)
  elseif(name STREQUAL "InklingMoEGate")
    list(APPEND sycl_install_args COMPONENT inkling_moe_gate)
  elseif(name STREQUAL "InklingRelativeAttention")
    list(APPEND sycl_install_args COMPONENT inkling_relative_attention)
  endif()
  install(TARGETS ${sycl_lib} ${sycl_install_args})
  set_target_properties(${sycl_lib} PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
endforeach()

# xe20 kernels
set(XE20_OFFLINE_COMPILER_AOT_OPTIONS "-device bmg")
set(XE20_OFFLINE_COMPILER_FLAGS "${XE20_OFFLINE_COMPILER_AOT_OPTIONS}${SYCL_OFFLINE_COMPILER_CG_OPTIONS}")
foreach(sycl_src ${ATen_XPU_SYCL_XE20})
  get_filename_component(name ${sycl_src} NAME_WLE REALPATH)
  set(sycl_lib sgl-ops-sycl-${name})
  sycl_add_library(
    ${sycl_lib}
    ${XE20_OFFLINE_COMPILER_FLAGS}
    ${COMMON_DEVICE_LINK_FLAGS}
    SHARED
    SYCL_SOURCES ${sycl_src})
  target_link_libraries(common_ops PUBLIC ${sycl_lib})
  list(APPEND SGL_OPS_LIBRARIES ${sycl_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_lib} LIBRARY DESTINATION sgl_kernel)
  set_target_properties(${sycl_lib} PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
endforeach()

if(TARGET sgl-ops-sycl-InklingSconv)
  Python3_add_library(
    inkling_sconv_ops
    MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI
    torch_extension_inkling_sconv.cc)
  install(TARGETS inkling_sconv_ops LIBRARY DESTINATION sgl_kernel COMPONENT inkling_sconv)
  set_target_properties(inkling_sconv_ops PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
  target_link_libraries(inkling_sconv_ops PUBLIC sgl-ops-sycl-InklingSconv)
  list(APPEND SGL_OPS_LIBRARIES inkling_sconv_ops)
endif()

if(TARGET sgl-ops-sycl-InklingAttnPrologue)
  Python3_add_library(
    inkling_attn_prologue_ops
    MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI
    torch_extension_inkling_attn_prologue.cc)
  install(TARGETS inkling_attn_prologue_ops LIBRARY DESTINATION sgl_kernel COMPONENT inkling_attn_prologue)
  set_target_properties(inkling_attn_prologue_ops PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
  target_link_libraries(inkling_attn_prologue_ops PUBLIC sgl-ops-sycl-InklingAttnPrologue)
  list(APPEND SGL_OPS_LIBRARIES inkling_attn_prologue_ops)
endif()

if(TARGET sgl-ops-sycl-InklingMoEGate)
  Python3_add_library(
    inkling_moe_gate_ops
    MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI
    torch_extension_inkling_moe_gate.cc)
  install(TARGETS inkling_moe_gate_ops LIBRARY DESTINATION sgl_kernel COMPONENT inkling_moe_gate)
  set_target_properties(inkling_moe_gate_ops PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
  target_link_libraries(inkling_moe_gate_ops PUBLIC sgl-ops-sycl-InklingMoEGate)
  list(APPEND SGL_OPS_LIBRARIES inkling_moe_gate_ops)
endif()

if(TARGET sgl-ops-sycl-InklingRelativeAttention)
  Python3_add_library(
    inkling_relative_attention_ops
    MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI
    torch_extension_inkling_relative_attention.cc)
  install(TARGETS inkling_relative_attention_ops LIBRARY DESTINATION sgl_kernel COMPONENT inkling_relative_attention)
  set_target_properties(inkling_relative_attention_ops PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
  target_link_libraries(inkling_relative_attention_ops PUBLIC sgl-ops-sycl-InklingRelativeAttention)
  list(APPEND SGL_OPS_LIBRARIES inkling_relative_attention_ops)
endif()

set(SYCL_LINK_LIBRARIES_KEYWORD)

foreach(lib ${SGL_OPS_LIBRARIES})
  # Align with PyTorch compile options PYTORCH_SRC_DIR/cmake/public/utils.cmake
  torch_compile_options(${lib})
  target_compile_options_if_supported(${lib} "-Wno-deprecated-copy")
  target_compile_options(${lib} PRIVATE ${TORCH_XPU_OPS_FLAGS})

  target_include_directories(${lib} PUBLIC ${TORCH_XPU_OPS_INCLUDE_DIRS})
  target_include_directories(${lib} PUBLIC ${ATen_XPU_INCLUDE_DIRS})
  target_include_directories(${lib} PUBLIC ${SYCL_INCLUDE_DIR})
  target_include_directories(${lib} PRIVATE ${Python3_INCLUDE_DIRS})
  target_include_directories(${lib} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${lib} PRIVATE ${Python3_LIBRARIES})

  target_include_directories(${lib} PRIVATE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(${lib} PRIVATE ${TORCH_LIBRARIES} c10 torch torch_cpu ${SYCL_LIBRARY})

  target_link_libraries(${lib} PUBLIC ${SYCL_LIBRARY})
endforeach()
