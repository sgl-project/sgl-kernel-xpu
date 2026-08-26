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
  set(COMMON_DEVICE_LINK_FLAGS ${COMMON_DEVICE_LINK_FLAGS} -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate)
else()
  message(FATAL_ERROR
      "SYCL compiler version must be >= 20250806, "
      "but got ${SYCL_COMPILER_VERSION}")
endif()

# common kernels
foreach(sycl_src ${ATen_XPU_SYCL_COMMON})
  get_filename_component(name ${sycl_src} NAME_WLE REALPATH)
  set(sycl_lib sgl-ops-sycl-${name})
  sycl_add_library(
    ${sycl_lib}
    ${SYCL_OFFLINE_COMPILER_FLAGS}
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

# xe20 kernels
set(XE20_OFFLINE_COMPILER_AOT_OPTIONS "-device bmg")
set(XE20_OFFLINE_COMPILER_FLAGS "${XE20_OFFLINE_COMPILER_AOT_OPTIONS}${SYCL_OFFLINE_COMPILER_CG_OPTIONS}")

# Instance families that are bundled into a single shared library each.
#
# The generated TUs within a family differ only in template arguments, so nearly
# all of their host-side CUTLASS/SYCL launch scaffolding is identical. Those
# instantiations are emitted as COMDAT groups, which a linker folds only *within
# one link*. Giving every TU its own .so therefore kept a private copy of that
# scaffolding per instance -- the 84 GroupGemmXe20 instances alone produced
# 227MB of .so for 1.5MB of actual device code, and per-file wheel compression
# cannot dedup across files. Bundling a family into one library lets COMDAT
# folding do its job without dropping a single AOT instance.
#
# Matched as a prefix against the source basename; a source matching no prefix
# gets its own library, as before.
set(SGL_XE20_BUNDLE_PREFIXES
  "GroupGemmXe20_inst_"
  "GroupGemmW4A16Xe20_inst_"
  "xe_fmha_fwd_decode_page_"
  "xe_fmha_fwd_decode_nopage_"
  "xe_fmha_fwd_split_decode_page_"
  "xe_fmha_fwd_prefill_page_"
  "xe_fmha_fwd_prefill_nopage_"
  "mla_decode_kernel_"
  "mla_prefill_kernel_"
  "mla_sparse_decode_kernel_"
  "mla_sparse_decode_2stage_kernel_"
  "mla_sparse_prefill_kernel_"
  "mla_sparse_prefill_2stage_kernel_"
  "sgemm_lora_a_fwd_kernel_"
  "sgemm_lora_b_fwd_kernel_"
)

# Pass 1: classify each xe20 source, recording the bundle key it belongs to (or
# leaving it empty for sources that keep their own library).
set(SGL_XE20_BUNDLE_KEYS)
foreach(sycl_src ${ATen_XPU_SYCL_XE20})
  get_filename_component(_src_name ${sycl_src} NAME_WLE REALPATH)
  set(_matched "")
  foreach(_prefix ${SGL_XE20_BUNDLE_PREFIXES})
    string(LENGTH "${_prefix}" _prefix_len)
    string(LENGTH "${_src_name}" _name_len)
    if(_name_len GREATER _prefix_len)
      string(SUBSTRING "${_src_name}" 0 ${_prefix_len} _head)
      if(_head STREQUAL "${_prefix}")
        set(_matched "${_prefix}")
        break()
      endif()
    endif()
  endforeach()

  if(_matched)
    # Strip trailing underscores so the library name reads as a family name.
    string(REGEX REPLACE "_+$" "" _key "${_matched}")
    if(NOT DEFINED SGL_XE20_BUNDLE_SRCS_${_key})
      set(SGL_XE20_BUNDLE_SRCS_${_key})
      list(APPEND SGL_XE20_BUNDLE_KEYS ${_key})
    endif()
    list(APPEND SGL_XE20_BUNDLE_SRCS_${_key} ${sycl_src})
    set(SGL_XE20_KEY_OF_${_src_name} ${_key})
  else()
    set(SGL_XE20_KEY_OF_${_src_name} "")
  endif()
endforeach()

# Emit the libraries, walking the original source order: a bundle is created at
# the position of its first member, everything else keeps its own library.
#
# Order matters and must match the unbundled layout. common_ops is linked with
# --as-needed, and a dispatcher library (e.g. GroupGemmXe20) only picks up its
# launcher instances if the providing library appears *after* it on the link
# line. The generated instances are appended to ATen_XPU_SYCL_XE20 last, after
# the hand-written dispatchers, so preserving that order keeps every bundle
# recorded as DT_NEEDED. Emitting bundles up front silently drops them.
set(SGL_XE20_EMITTED_KEYS)
foreach(sycl_src ${ATen_XPU_SYCL_XE20})
  get_filename_component(_src_name ${sycl_src} NAME_WLE REALPATH)
  set(_key "${SGL_XE20_KEY_OF_${_src_name}}")

  if(_key)
    # Bundled source: emit the whole family the first time we reach one of it.
    if(_key IN_LIST SGL_XE20_EMITTED_KEYS)
      continue()
    endif()
    list(APPEND SGL_XE20_EMITTED_KEYS ${_key})
    set(sycl_lib sgl-ops-sycl-${_key})
    set(_lib_srcs ${SGL_XE20_BUNDLE_SRCS_${_key}})
    list(LENGTH _lib_srcs _n_srcs)
    message(STATUS "Bundling ${_n_srcs} xe20 instances into ${sycl_lib}")
  else()
    set(sycl_lib sgl-ops-sycl-${_src_name})
    set(_lib_srcs ${sycl_src})
  endif()

  sycl_add_library(
    ${sycl_lib}
    ${XE20_OFFLINE_COMPILER_FLAGS}
    ${COMMON_DEVICE_LINK_FLAGS}
    SHARED
    SYCL_SOURCES ${_lib_srcs})
  target_link_libraries(common_ops PUBLIC ${sycl_lib})
  list(APPEND SGL_OPS_LIBRARIES ${sycl_lib})

  # Decouple with PyTorch cmake definition.
  install(TARGETS ${sycl_lib} LIBRARY DESTINATION sgl_kernel)
  set_target_properties(${sycl_lib} PROPERTIES
    INSTALL_RPATH "$ORIGIN"
    BUILD_WITH_INSTALL_RPATH TRUE
  )
endforeach()

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
