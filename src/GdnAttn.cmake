# GdnAttn.cmake — vendored from vllm-xpu-kernels (gdn_attention op + Xe2 chunked
# GDR fast path). Self-contained so it never has to touch upstream's
# src/sycl/* glob or FindSYCL paths.
#
# Files:
#   src/gdn_attn/gdn_attn_interface.cpp        — dispatch / schema host code
#   src/gdn_attn/torch_bindings_gdn_attn.cc    — TORCH_LIBRARY_FRAGMENT(sgl_kernel)
#   src/gdn_attn/causal_conv1d.hpp,
#       gated_delta_rule.hpp,
#       gdn_attn_utils.h,
#       vllm_xpu_utils.h, vllm_xpu_dispatch_utils.h  — headers
#   src/gdn_attn/xe_2/chunk_gated_delta_rule_xe2Xe20.cpp — BMG AOT kernel
#   src/gdn_attn/xe_2/*.hpp, gemm.hpp                   — Xe2 kernel headers
#
# All .cpp files here are standard SYCL (no ESIMD), so we can use the upstream
# sycl_add_library() helper with the same flag set as other sycl/*.cpp files.
# The Xe20-suffixed file gets AOT-compiled for BMG via the XE20 flag bundle.

set(GDN_ATTN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gdn_attn)

file(GLOB GDN_ATTN_HOST_SRCS
    "${GDN_ATTN_DIR}/*.cc")

file(GLOB GDN_ATTN_COMMON_SRCS
    "${GDN_ATTN_DIR}/*.cpp")

file(GLOB GDN_ATTN_XE20_SRCS
    "${GDN_ATTN_DIR}/xe_2/*Xe20.cpp")

# Attach host-side .cc (torch_bindings) to common_ops directly; it's plain C++.
target_sources(common_ops PRIVATE ${GDN_ATTN_HOST_SRCS})

# Make sure the header directory and VLLM_XPU_ENABLE_XE2 reach every translation
# unit in the main target, so gdn_attn_interface.cpp picks the Xe2 fast path.
target_include_directories(common_ops PRIVATE
    ${GDN_ATTN_DIR}
    ${GDN_ATTN_DIR}/xe_2)
# gdn_attn_interface.cpp + chunk_gated_delta_rule_xe2Xe20.cpp both rely on
# these macros (the cutlass-sycl headers gate a lot of symbols behind
# SYCL_INTEL_TARGET, including `cutlass::get_sub_group_id` in cutlass.h).
# Mirrors the flags that vllm-xpu-kernels passes when compiling the same files.
target_compile_definitions(common_ops PRIVATE
    VLLM_XPU_ENABLE_XE2
    SYCL_INTEL_TARGET
    CUTLASS_ENABLE_HEADERS_ONLY
    CUTLASS_VERSIONS_GENERATED)

# Compile the non-Xe20 host SYCL sources with the upstream common flag set.
foreach(sycl_src ${GDN_ATTN_COMMON_SRCS})
  get_filename_component(name ${sycl_src} NAME_WLE REALPATH)
  set(sycl_lib sgl-ops-sycl-${name})
  sycl_add_library(
    ${sycl_lib}
    ${SYCL_OFFLINE_COMPILER_FLAGS}
    ${COMMON_DEVICE_LINK_FLAGS}
    SHARED
    SYCL_SOURCES ${sycl_src})
  target_include_directories(${sycl_lib} PRIVATE
    ${GDN_ATTN_DIR}
    ${GDN_ATTN_DIR}/xe_2
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${SYCL_INCLUDE_DIR})
  target_compile_definitions(${sycl_lib} PRIVATE
      VLLM_XPU_ENABLE_XE2
      SYCL_INTEL_TARGET
      CUTLASS_ENABLE_HEADERS_ONLY
      CUTLASS_VERSIONS_GENERATED)
  torch_compile_options(${sycl_lib})
  target_compile_options(${sycl_lib} PRIVATE ${TORCH_XPU_OPS_FLAGS})
  # NOTE: sycl_add_library() calls target_link_libraries in plain mode, so we
  # must stay in plain mode here too (CMake forbids mixing plain + keyword).
  target_link_libraries(${sycl_lib}
    ${TORCH_LIBRARIES} c10 torch torch_cpu ${SYCL_LIBRARY})
  target_link_libraries(common_ops PUBLIC ${sycl_lib})
  install(TARGETS ${sycl_lib} LIBRARY DESTINATION sgl_kernel)
  set_target_properties(${sycl_lib} PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE)
endforeach()

# Xe20 (BMG AOT) sources use the same XE20_OFFLINE_COMPILER_FLAGS as upstream.
foreach(sycl_src ${GDN_ATTN_XE20_SRCS})
  get_filename_component(name ${sycl_src} NAME_WLE REALPATH)
  set(sycl_lib sgl-ops-sycl-${name})
  sycl_add_library(
    ${sycl_lib}
    ${XE20_OFFLINE_COMPILER_FLAGS}
    ${COMMON_DEVICE_LINK_FLAGS}
    SHARED
    SYCL_SOURCES ${sycl_src})
  target_include_directories(${sycl_lib} PRIVATE
    ${GDN_ATTN_DIR}
    ${GDN_ATTN_DIR}/xe_2
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${SYCL_INCLUDE_DIR})
  target_compile_definitions(${sycl_lib} PRIVATE
      VLLM_XPU_ENABLE_XE2
      SYCL_INTEL_TARGET
      CUTLASS_ENABLE_HEADERS_ONLY
      CUTLASS_VERSIONS_GENERATED)
  torch_compile_options(${sycl_lib})
  target_compile_options(${sycl_lib} PRIVATE ${TORCH_XPU_OPS_FLAGS})
  # NOTE: sycl_add_library() calls target_link_libraries in plain mode, so we
  # must stay in plain mode here too (CMake forbids mixing plain + keyword).
  target_link_libraries(${sycl_lib}
    ${TORCH_LIBRARIES} c10 torch torch_cpu ${SYCL_LIBRARY})
  target_link_libraries(common_ops PUBLIC ${sycl_lib})
  install(TARGETS ${sycl_lib} LIBRARY DESTINATION sgl_kernel)
  set_target_properties(${sycl_lib} PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE)
endforeach()
