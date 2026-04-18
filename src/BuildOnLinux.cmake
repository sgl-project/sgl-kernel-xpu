# Build on Linux

set(SGL_OPS_LIBRARIES)
set(SYCL_LINK_LIBRARIES_KEYWORD PRIVATE)

# AOT device link flags, shared by every SYCL kernel library below and reused
# verbatim by the runtime-JIT compile so both paths link identically.
if(SYCL_COMPILER_VERSION GREATER_EQUAL 20250806)
  # SYCL_DEVICE_LINK_FLAGS already contains SGL_SYCL_SPIRV_EXT_FLAGS
  # (appended in cmake/BuildFlags.cmake); do not append it again.
  set(COMMON_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS})
else()
  message(FATAL_ERROR
      "SYCL compiler version must be >= 20250806, "
      "but got ${SYCL_COMPILER_VERSION}")
endif()

# Runtime-JIT engine (pure host C++, depends only on libdl). Linked into the
# SYCL libraries whose dispatch renders/compiles the *.cpp.in templates on
# demand (flash_attention for FMHA, GroupGemmXe20 for MoE grouped GEMM).
if(USE_SYCL_JIT)
  add_library(sgl_jit STATIC
    ${SGL_OPS_XPU_ROOT}/src/jit/sycl_template_jit.cpp
    ${SGL_OPS_XPU_ROOT}/src/jit/jit_arch.cpp
    ${SGL_OPS_XPU_ROOT}/src/jit/fmha_jit.cpp
    ${SGL_OPS_XPU_ROOT}/src/jit/moe_jit.cpp
    ${SGL_OPS_XPU_ROOT}/src/jit/mla_jit.cpp
    ${SGL_OPS_XPU_ROOT}/src/jit/gdn_jit.cpp)
  set_target_properties(sgl_jit PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(sgl_jit PUBLIC ${SGL_OPS_XPU_ROOT}/src)
  # Feed the runtime-JIT compile the exact AOT flags: host flags (-fPIC,
  # -std=c++20, visibility, warnings) + kernel compile options + shared
  # device-link flags, so a JIT-compiled kernel is byte-for-byte equivalent to
  # its AOT sibling. -shared makes the single-shot compile emit a .so;
  # -DCUTLASS_ENABLE_SYCL is added on top because the AOT build supplies it
  # globally via add_compile_definitions (not part of SYCL_COMPILE_FLAGS) and
  # without it cutlass-sycl headers take the CUDA path and fail on
  # <cuda_runtime_api.h>. default_sycl_flags() drops the -fsycl-targets token so
  # the per-arch JIT target wins.
  string(REPLACE ";" " " SGL_JIT_SYCL_FLAGS_VALUE
    "-shared;${SYCL_HOST_FLAGS};${SYCL_COMPILE_FLAGS};${COMMON_DEVICE_LINK_FLAGS};-DCUTLASS_ENABLE_SYCL")
  # AOT feeds the IGC/ocloc backend the -cl-* codegen options via `-Xs` at the
  # device-link step (SYCL_OFFLINE_COMPILER_CG_OPTIONS). The single-shot JIT must
  # pass the same options (e.g. correctly-rounded fp32 div/sqrt, >4GB buffers,
  # auto large-GRF) or its device code diverges from the AOT sibling. Kept as a
  # separate macro: it is one `-Xs` argument and must NOT be whitespace-split
  # like SGL_JIT_SYCL_FLAGS. The AOT `-device` selector is omitted because the
  # per-arch -fsycl-targets alias already fixes the JIT device.
  string(STRIP "${SYCL_OFFLINE_COMPILER_CG_OPTIONS}" SGL_JIT_XS_FLAGS_VALUE)
  target_compile_definitions(sgl_jit PRIVATE
    SGL_JIT_SYCL_FLAGS=\"${SGL_JIT_SYCL_FLAGS_VALUE}\"
    SGL_JIT_XS_FLAGS=\"${SGL_JIT_XS_FLAGS_VALUE}\")
  target_compile_features(sgl_jit PRIVATE cxx_std_17)
  target_link_libraries(sgl_jit PUBLIC ${CMAKE_DL_LIBS})
endif()

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

# Dispatchers that call the runtime-JIT engine link the static JIT library.
if(USE_FMHA AND USE_SYCL_JIT AND TARGET sgl-ops-sycl-flash_attention)
  target_link_libraries(sgl-ops-sycl-flash_attention PRIVATE sgl_jit)
endif()
if(USE_MLA AND USE_SYCL_JIT AND TARGET sgl-ops-sycl-mla_decode)
  target_link_libraries(sgl-ops-sycl-mla_decode PRIVATE sgl_jit)
endif()
if(USE_MLA AND USE_SYCL_JIT AND TARGET sgl-ops-sycl-mla_prefill)
  target_link_libraries(sgl-ops-sycl-mla_prefill PRIVATE sgl_jit)
endif()
if(USE_MLA AND USE_SYCL_JIT AND TARGET sgl-ops-sycl-mla_sparse_decode)
  target_link_libraries(sgl-ops-sycl-mla_sparse_decode PRIVATE sgl_jit)
endif()
if(USE_MLA AND USE_SYCL_JIT AND TARGET sgl-ops-sycl-mla_sparse_prefill)
  target_link_libraries(sgl-ops-sycl-mla_sparse_prefill PRIVATE sgl_jit)
endif()

# xe20 kernels
# The AOT device must match the arch these sources were *compiled* for: CMakeLists.txt
# applies SYCL_INTEL_TARGET globally, so on a cri build these TUs emit Xe3p vISA and
# ocloc -device bmg rejects it ("platform requires 2 (l1-l3) caching options").
set(XE20_OFFLINE_COMPILER_AOT_OPTIONS "${SYCL_OFFLINE_COMPILER_AOT_OPTIONS}")
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

# The bf16 grouped GEMM dispatch in GroupGemmXe20.cpp calls the runtime-JIT engine.
if(USE_MOE AND USE_SYCL_JIT AND TARGET sgl-ops-sycl-GroupGemmXe20)
  target_link_libraries(sgl-ops-sycl-GroupGemmXe20 PRIVATE sgl_jit)
endif()
# The W4A16 grouped GEMM dispatch in GroupGemmW4A16Xe20.cpp calls the JIT engine.
if(USE_MOE AND USE_SYCL_JIT AND TARGET sgl-ops-sycl-GroupGemmW4A16Xe20)
  target_link_libraries(sgl-ops-sycl-GroupGemmW4A16Xe20 PRIVATE sgl_jit)
endif()
# The GDN chunk delta-rule dispatch (chunk_gated_delta_rule.cpp) calls the JIT engine.
if(USE_FMHA AND USE_SYCL_JIT AND TARGET sgl-ops-sycl-chunk_gated_delta_rule)
  target_link_libraries(sgl-ops-sycl-chunk_gated_delta_rule PRIVATE sgl_jit)
endif()

# xe35 kernels (CRI only)
if(DPCPP_SYCL_TARGET MATCHES "cri")
  set(XE35_OFFLINE_COMPILER_AOT_OPTIONS "-device cri")
  set(XE35_OFFLINE_COMPILER_FLAGS "${XE35_OFFLINE_COMPILER_AOT_OPTIONS}${SYCL_OFFLINE_COMPILER_CG_OPTIONS}")
  foreach(sycl_src ${ATen_XPU_SYCL_XE35})
    get_filename_component(name ${sycl_src} NAME_WLE REALPATH)
    set(sycl_lib sgl-ops-sycl-${name})
    sycl_add_library(
      ${sycl_lib}
      ${XE35_OFFLINE_COMPILER_FLAGS}
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
  target_link_libraries(${lib} PRIVATE ${Python3_LIBRARIES})

  target_include_directories(${lib} PRIVATE ${TORCH_INCLUDE_DIRS})
  target_link_libraries(${lib} PRIVATE ${TORCH_LIBRARIES} c10 torch torch_cpu ${SYCL_LIBRARY})

  target_link_libraries(${lib} PUBLIC ${SYCL_LIBRARY})
endforeach()
