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

# Variant of the offline-compiler flags that forces 256 GRF per thread for
# register-heavy kernels (e.g. FMHA prefill). The default auto-large-GRF mode
# lets IGC pick 128 GRF for these kernels, which causes register spills and a
# measurable performance loss versus the 256-GRF build.
string(REPLACE
  "-cl-intel-enable-auto-large-GRF-mode"
  "-cl-intel-256-GRF-per-thread"
  SYCL_OFFLINE_COMPILER_FLAGS_256GRF
  "${SYCL_OFFLINE_COMPILER_FLAGS}")

# common kernels
foreach(sycl_src ${ATen_XPU_SYCL_COMMON})
  get_filename_component(name ${sycl_src} NAME_WLE REALPATH)
  set(sycl_lib sgl-ops-sycl-${name})
  if(name MATCHES "xe_fmha_fwd_prefill_kernel")
    set(sycl_offline_flags ${SYCL_OFFLINE_COMPILER_FLAGS_256GRF})
  else()
    set(sycl_offline_flags ${SYCL_OFFLINE_COMPILER_FLAGS})
  endif()
  sycl_add_library(
    ${sycl_lib}
    ${sycl_offline_flags}
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
