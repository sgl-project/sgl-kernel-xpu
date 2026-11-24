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
  set(COMMON_DEVICE_LINK_FLAGS ${COMMON_DEVICE_LINK_FLAGS} -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate,+SPV_INTEL_float4)
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

# xe20 kernels (BMG only)
if(DPCPP_SYCL_TARGET MATCHES "bmg")
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
endif()

# xe35 kernels (CRI only)
if(DPCPP_SYCL_TARGET MATCHES "cri")
  set(XE35_OFFLINE_COMPILER_AOT_OPTIONS "-device cri")
  set(XE35_OFFLINE_COMPILER_FLAGS "${XE35_OFFLINE_COMPILER_AOT_OPTIONS}${SYCL_OFFLINE_COMPILER_CG_OPTIONS}")

  # GEMM kernels get a different GRF (thread register file) mode from the rest.
  #
  # A GEMM mainloop holds a large FP32 accumulator live across the whole k-loop,
  # plus operand fragments and any dequant/scale state, so it spills in the
  # 256-GRF large mode that IGC's auto heuristic picks. CRI has a 512-GRF thread
  # mode that removes those spills. Element-wise and other latency-bound kernels
  # keep the original auto-large-GRF flags: they carry little live state and would
  # rather have the threads-per-EU that 512 GRF halves. Element-wise kernels are
  # built by the "common kernels" loop above, which is left untouched.
  #
  # Measured on the NVFP4 WoQ mainloops (IGC asm dumps, k-tile loop body):
  #
  #                    256 GRF (auto)              512 GRF
  #   MXFP8 WoQ    12,736 B spill, 651 sends    no spill, 114 sends
  #   BF16  WoQ     7,296 B spill, 300 sends    no spill, 104 sends
  #
  # The sends removed are scratch-memory spill/fill traffic in the innermost loop.
  #
  # Matched against the source basename (no extension) so that the generated
  # instantiation TUs (GroupGemm*_inst_xe35__*, sgemm_lora_a_fwd_kernel_*) are
  # covered along with their parent kernel. Attention (flash_attention*,
  # mla_decode_*) is intentionally excluded: it is register-heavy too, but it is
  # occupancy-sensitive and separately tuned, so it needs its own measurement
  # before being moved.
  set(XE35_GEMM_KERNEL_PATTERNS
      "^blockwise_moe_"              # MoE grouped GEMMs (mxfp4, mxfp8, nvfp4_woq)
      "_gemm$"                       # dsv3_fused_a_gemm, dsv3_router_gemm, fp8_blockwise_gemm
      "^fp8_scaled_mm$"
      "^GroupGemm"                   # GroupGemm, GroupGemmMxfp4W4A16 + generated instantiations
      "^sgemm_lora_a_fwd_kernel_")   # LoRA A-projection GEMM instantiations
  string(REPLACE
      "${SYCL_GRF_MODE_OPTION}" "${SYCL_GRF_MODE_512_OPTION}"
      XE35_CG_OPTIONS_GEMM "${SYCL_OFFLINE_COMPILER_CG_OPTIONS}")
  set(XE35_OFFLINE_COMPILER_FLAGS_GEMM
      "${XE35_OFFLINE_COMPILER_AOT_OPTIONS}${XE35_CG_OPTIONS_GEMM}")

  foreach(sycl_src ${ATen_XPU_SYCL_XE35})
    get_filename_component(name ${sycl_src} NAME_WLE REALPATH)
    set(sycl_lib sgl-ops-sycl-${name})
    set(_xe35_is_gemm FALSE)
    foreach(_pat ${XE35_GEMM_KERNEL_PATTERNS})
      if(name MATCHES "${_pat}")
        set(_xe35_is_gemm TRUE)
        break()
      endif()
    endforeach()
    if(_xe35_is_gemm)
      set(_xe35_offline_flags "${XE35_OFFLINE_COMPILER_FLAGS_GEMM}")
      list(APPEND XE35_GEMM_512GRF_KERNELS ${name})
    else()
      set(_xe35_offline_flags "${XE35_OFFLINE_COMPILER_FLAGS}")
      list(APPEND XE35_DEFAULT_GRF_KERNELS ${name})
    endif()
    sycl_add_library(
      ${sycl_lib}
      ${_xe35_offline_flags}
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
  list(LENGTH XE35_GEMM_512GRF_KERNELS _n_gemm)
  list(LENGTH XE35_DEFAULT_GRF_KERNELS _n_other)
  message(STATUS "CRI GRF modes: ${_n_gemm} GEMM kernels at 512 GRF, "
                 "${_n_other} kernels at default auto-large-GRF")
  message(STATUS "CRI default-GRF kernels: ${XE35_DEFAULT_GRF_KERNELS}")
endif()

# xe40 kernels (JGS only)
if(DPCPP_SYCL_TARGET MATCHES "jgs")
  set(XE40_OFFLINE_COMPILER_AOT_OPTIONS "-device jgs-a0 -options -ze-intel-xe-features=+set-abarrier-arrive-lmc,+disable-misched,+total-grf-num-96,+null-dst")
  set(XE40_OFFLINE_COMPILER_FLAGS "${XE40_OFFLINE_COMPILER_AOT_OPTIONS}${SYCL_OFFLINE_COMPILER_CG_OPTIONS}")
  set(XE40_TARGETS_OPTION -fsycl-targets=pisa)
  set(XE40_TARGETS_OPTION ${XE40_TARGETS_OPTION} --offload-new-driver)
  set(XE40_TARGETS_OPTION ${XE40_TARGETS_OPTION} --offload-arch=jgs)
  set(XE40_DEVICE_LINK_FLAGS ${SYCL_DEVICE_LINK_FLAGS} "-fsycl-targets=spir64_gen")
  set(XE40_DEVICE_LINK_FLAGS ${XE40_DEVICE_LINK_FLAGS} ${XE40_TARGETS_OPTION})
  # WA for NEO-16141
  set(XE40_TARGETS_OPTION ${XE40_TARGETS_OPTION} --pisa-fin-path=$ENV{CMPLR_ROOT}/bin/compiler)
  foreach(sycl_src ${ATen_XPU_SYCL_XE40})
    get_filename_component(name ${sycl_src} NAME_WLE REALPATH)
    set(sycl_lib sgl-ops-sycl-${name})
    sycl_add_library(
      ${sycl_lib}
      ${XE40_OFFLINE_COMPILER_FLAGS}
      ${XE40_DEVICE_LINK_FLAGS}
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
