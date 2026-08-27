# ATen XPU sources

# Feature toggles. When a feature is turned OFF, the corresponding kernels are
# neither compiled (their source/generated files are filtered out below) nor
# registered as torch ops (guarded by matching macros in torch_extension_sycl.cc).
#   USE_MOE  -> MoE kernels
#   USE_FMHA -> flash attention (fwd / mha_fwd)
#   USE_GDN  -> Gated DeltaNet attention kernels
#   USE_MLA  -> MLA decode/prefill/sparse-decode kernels
#   USE_MLA_SPARSE_FUSED -> fused (single-pass) sparse-MLA-decode kernel (optimization track)
#
# Defaults are set in the option() declarations below. Each toggle can be set either as a CMake option
# (-DUSE_MOE=OFF) or, for convenience with `pip install`, via an environment
# variable of the same name (e.g. `USE_MOE=OFF pip install -v .` or
# `USE_MOE=0 pip install -v .`). Both `OFF` and `0` (case-insensitive) disable a
# feature; the environment variable takes precedence when set.
option(USE_MOE "Build MoE kernels" ON)
option(USE_FMHA "Build flash attention kernels" ON)
option(USE_GDN "Build Gated DeltaNet attention kernels" ON)
option(USE_MLA "Build MLA kernels" ON)
# The fused (single-pass) sparse-MLA-decode kernel is the optimization track; the
# 2-stage kernel is the shipping runtime default. Building the fused variant is
# opt-in so default/CI builds don't pay for its extra instantiations. Turning it
# OFF also pins the runtime dispatch to the 2-stage path (see below) so the fused
# launchers are never referenced.
option(USE_MLA_SPARSE_FUSED "Build the fused (single-pass) sparse MLA decode kernel (optimization track)" OFF)
# Master runtime-JIT switch (opt-in, default OFF). When ON, kernels that support
# runtime codegen (currently FMHA decode/prefill) are compiled on demand via
# src/jit instead of being instantiated ahead of time. Enable with USE_SYCL_JIT=ON
# (or 1 / On) as a CMake option or environment variable.
option(USE_SYCL_JIT "Enable runtime-JIT kernels (opt-in; drops their AOT .so instances)" OFF)

foreach(_toggle USE_MOE USE_FMHA USE_GDN USE_MLA USE_MLA_SPARSE_FUSED USE_SYCL_JIT)
    if(DEFINED ENV{${_toggle}} AND NOT "$ENV{${_toggle}}" STREQUAL "")
        # Normalize the env value so 1 / On / ON / on / true / yes all enable and
        # 0 / Off / OFF / false / no disable, independent of CMake's if(<string>)
        # constant quirks.
        string(TOUPPER "$ENV{${_toggle}}" _toggle_val)
        if(_toggle_val MATCHES "^(1|ON|YES|TRUE|Y)$")
            set(${_toggle} ON)
        elseif(_toggle_val MATCHES "^(0|OFF|NO|FALSE|N)$")
            set(${_toggle} OFF)
        else()
            message(FATAL_ERROR
                "Invalid value for ${_toggle}: '$ENV{${_toggle}}' "
                "(expected one of 1/ON/On/true/yes or 0/OFF/Off/false/no)")
        endif()
    endif()
endforeach()

message(STATUS "USE_MOE=${USE_MOE}")
message(STATUS "USE_FMHA=${USE_FMHA}")
message(STATUS "USE_GDN=${USE_GDN}")
message(STATUS "USE_MLA=${USE_MLA}")
message(STATUS "USE_MLA_SPARSE_FUSED=${USE_MLA_SPARSE_FUSED}")
message(STATUS "USE_SYCL_JIT=${USE_SYCL_JIT}")

if(USE_MOE)
    add_compile_definitions(USE_MOE)
endif()
if(USE_FMHA)
    add_compile_definitions(USE_FMHA)
endif()
if(USE_GDN)
    add_compile_definitions(USE_GDN)
endif()
if(USE_FMHA AND USE_SYCL_JIT)
    # Route decode/prefill dispatch (flash_attention.cpp) through the runtime-JIT
    # path; the AOT FMHA instances are not generated (see include() guard below).
    add_compile_definitions(USE_FMHA_JIT)
endif()
if(USE_MOE AND USE_SYCL_JIT)
    # Route the bf16 grouped GEMM dispatch (GroupGemmXe20.cpp) through the
    # runtime-JIT path; its AOT instances are not generated.
    add_compile_definitions(USE_MOE_JIT)
endif()
if(USE_MLA AND USE_SYCL_JIT)
    # Route the MLA decode dispatch (mla_decode.cpp) through the runtime-JIT path.
    add_compile_definitions(USE_MLA_JIT)
endif()
if(USE_FMHA AND USE_SYCL_JIT)
    # Route the GDN chunk delta-rule dispatch (chunk_gated_delta_rule) through the
    # runtime-JIT path; its per-dtype kernel_launcher instances are not compiled
    # ahead of time (GDN shares the USE_FMHA toggle).
    add_compile_definitions(USE_GDN_JIT)
endif()
if(USE_MLA)
    add_compile_definitions(USE_MLA)
endif()
if(USE_MLA_SPARSE_FUSED)
    add_compile_definitions(USE_MLA_SPARSE_FUSED)
else()
    # Fused kernel not built -> force the runtime dispatch to the 2-stage path so
    # mla_sparse_decode.cpp never references the (uncompiled) fused launchers.
    add_compile_definitions(SGLANG_USE_SPARSE_MLA_2STAGE=1)
endif()

file(GLOB device_cpp "sycl/*.cpp" "sycl/*.sycl")
file(GLOB host_cpp "./*.cpp" "./*.cc")

# Filter out source files belonging to disabled features BEFORE splitting them
# into common/Xe20 buckets. The disabled base sources (flash_attention.cpp,
# mla_*.cpp) are non-Xe20 and would otherwise land in device_cpp_common, which
# is never filtered; removing them from device_cpp up front is the only way to
# keep them out of the build.
if(NOT USE_FMHA)
    list(FILTER device_cpp EXCLUDE REGEX "/flash_attention\\.cpp$")
endif()
if(NOT USE_GDN)
    list(FILTER device_cpp EXCLUDE REGEX "/gdn_attention_Xe20\\.cpp$")
    list(FILTER device_cpp EXCLUDE REGEX "/gdn_attn_workspace\\.cpp$")
endif()
if(NOT USE_MLA)
    list(FILTER device_cpp EXCLUDE REGEX "/mla_(decode|prefill|sparse_decode)\\.cpp$")
endif()
if(NOT USE_MOE)
    list(FILTER device_cpp EXCLUDE REGEX "/(GroupGemm(Xe20|W4A16Xe20|Fp8W8A16Xe20)|MoE(Align|PrepareInputs|Sum|_fused_gate|_sum_reduce))\\.cpp$")
endif()

# BMG only
set(device_cpp_xe20)
# common SYCL kernels
set(device_cpp_common)

foreach(file ${device_cpp})
    get_filename_component(fname ${file} NAME)

    if(fname MATCHES "Xe20\\.(cpp|sycl)$")
        list(APPEND device_cpp_xe20 ${file})
    else()
        list(APPEND device_cpp_common ${file})
    endif()

endforeach()

include(${CMAKE_CURRENT_SOURCE_DIR}/SGEMMLoraAFwdXe20.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/SGEMMLoraBFwdXe20.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/QKVLoraBFwdXe20.cmake)
if(USE_FMHA)
    if(NOT USE_SYCL_JIT)
        include(FMHADecodeXe20.cmake)
        include(FMHAPrefillXe20.cmake)
    endif()
endif()
if(USE_GDN)
    include(GdnAttnXe20.cmake)
endif()
if(USE_MLA)
    if(NOT USE_SYCL_JIT)
        include(MlaDecodeXe20.cmake)
        include(MlaPrefillXe20.cmake)
        include(MlaSparseDecodeXe20.cmake)
        include(MlaSparsePrefillXe20.cmake)
    endif()
endif()
if(USE_MOE)
    if(NOT USE_SYCL_JIT)
        include(${CMAKE_CURRENT_SOURCE_DIR}/GroupGemmXe20.cmake)
        include(${CMAKE_CURRENT_SOURCE_DIR}/GroupGemmW4A16Xe20.cmake)
        include(${CMAKE_CURRENT_SOURCE_DIR}/GroupGemmFp8W8A16Xe20.cmake)
    endif()
endif()


message(STATUS "BMG files: ${device_cpp_xe20}")
message(STATUS "Common files: ${device_cpp_common}")

# Ship the headers + templates the runtime-JIT kernels need to compile: the sgl
# source-header tree (so quote-includes like "sycl/kernels/..." resolve) plus the
# *.cpp.in templates and the CUTLASS-SYCL headers. These are text (highly
# compressible) and replace the precompiled FMHA / grouped-GEMM .so instances.
if(USE_SYCL_JIT)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/sycl
            DESTINATION sgl_kernel/include/sgl_src
            FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h" PATTERN "*.cpp.in")
    # sycl/Utils.h quote-includes "jit/jit_arch.h"; ship the JIT host headers so
    # they resolve against the sgl_src include root during the on-demand compile.
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/jit
            DESTINATION sgl_kernel/include/sgl_src
            FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h")
    if(DEFINED repo-cutlass-sycl_SOURCE_DIR)
        install(DIRECTORY ${repo-cutlass-sycl_SOURCE_DIR}/include/
                DESTINATION sgl_kernel/include/cutlass_sycl/include)
        install(DIRECTORY ${repo-cutlass-sycl_SOURCE_DIR}/tools/util/include/
                DESTINATION sgl_kernel/include/cutlass_sycl/tools/util/include)
    else()
        message(WARNING "repo-cutlass-sycl_SOURCE_DIR not set; FMHA JIT headers "
                        "will not be packaged. Set SGL_JIT_CUTLASS_INCLUDE at runtime.")
    endif()
endif()

list(APPEND ATen_XPU_CPP_SRCS ${host_cpp})
list(APPEND ATen_XPU_SYCL_COMMON ${device_cpp_common})
list(APPEND ATen_XPU_SYCL_XE20 ${device_cpp_xe20})
set(ATen_XPU_CPP_SRCS ${ATen_XPU_CPP_SRCS} PARENT_SCOPE)
set(ATen_XPU_SYCL_COMMON ${ATen_XPU_SYCL_COMMON} PARENT_SCOPE)
set(ATen_XPU_SYCL_XE20 ${ATen_XPU_SYCL_XE20} PARENT_SCOPE)

include(BuildOnLinux.cmake)
