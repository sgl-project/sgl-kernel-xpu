# ATen XPU sources

# Feature toggles. When a feature is turned OFF, the corresponding kernels are
# neither compiled (their source/generated files are filtered out below) nor
# registered as torch ops (guarded by matching macros in torch_extension_sycl.cc).
#   USE_MOE  -> MoE kernels
#   USE_FMHA -> flash attention (fwd / mha_fwd)
#   USE_MLA  -> MLA decode/prefill/sparse-decode kernels
#
# All default to ON. Each toggle can be set either as a CMake option
# (-DUSE_MOE=OFF) or, for convenience with `pip install`, via an environment
# variable of the same name (e.g. `USE_MOE=OFF pip install -v .` or
# `USE_MOE=0 pip install -v .`). Both `OFF` and `0` (case-insensitive) disable a
# feature; the environment variable takes precedence when set.
option(USE_MOE "Build MoE kernels" ON)
option(USE_FMHA "Build flash attention kernels" ON)
option(USE_MLA "Build MLA kernels" ON)

foreach(_toggle USE_MOE USE_FMHA USE_MLA)
    if(DEFINED ENV{${_toggle}} AND NOT "$ENV{${_toggle}}" STREQUAL "")
        if("$ENV{${_toggle}}")
            set(${_toggle} ON)
        else()
            set(${_toggle} OFF)
        endif()
    endif()
endforeach()

message(STATUS "USE_MOE=${USE_MOE}")
message(STATUS "USE_FMHA=${USE_FMHA}")
message(STATUS "USE_MLA=${USE_MLA}")

if(USE_MOE)
    add_compile_definitions(USE_MOE)
endif()
if(USE_FMHA)
    add_compile_definitions(USE_FMHA)
endif()
if(USE_MLA)
    add_compile_definitions(USE_MLA)
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
if(NOT USE_MLA)
    list(FILTER device_cpp EXCLUDE REGEX "/mla_(decode|prefill|sparse_decode)\\.cpp$")
endif()
if(NOT USE_MOE)
    list(FILTER device_cpp EXCLUDE REGEX "/(GroupGemm(Xe20|Mxfp4W4A16Xe20)|MoE(Align|PrepareInputs|Sum|_fused_gate|_sum_reduce))\\.cpp$")
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

include(GdnAttnXe20.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/SGEMMLoraAFwdXe20.cmake)
if(USE_FMHA)
    include(FMHADecodeXe20.cmake)
    include(FMHAPrefillXe20.cmake)
endif()
if(USE_MLA)
    include(MlaDecodeXe20.cmake)
    include(MlaPrefillXe20.cmake)
    include(MlaSparseDecodeXe20.cmake)
endif()
if(USE_MOE)
    include(${CMAKE_CURRENT_SOURCE_DIR}/GroupGemmXe20.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/GroupGemmMxfp4W4A16Xe20.cmake)
endif()

message(STATUS "BMG files: ${device_cpp_xe20}")
message(STATUS "Common files: ${device_cpp_common}")

list(APPEND ATen_XPU_CPP_SRCS ${host_cpp})
list(APPEND ATen_XPU_SYCL_COMMON ${device_cpp_common})
list(APPEND ATen_XPU_SYCL_XE20 ${device_cpp_xe20})

set(ATen_XPU_CPP_SRCS ${ATen_XPU_CPP_SRCS} PARENT_SCOPE)
set(ATen_XPU_SYCL_COMMON ${ATen_XPU_SYCL_COMMON} PARENT_SCOPE)
set(ATen_XPU_SYCL_XE20 ${ATen_XPU_SYCL_XE20} PARENT_SCOPE)

include(BuildOnLinux.cmake)
