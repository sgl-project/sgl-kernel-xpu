# ATen XPU sources

file(GLOB device_cpp "sycl/*.cpp" "sycl/*.sycl")
file(GLOB host_cpp "./*.cpp" "./*.cc")

# List of kernels that should only be built for CRI target.
# Add kernel filenames (without path) to this list as needed.
set(CRI_ONLY_KERNELS
  blockwise_moe_kernel.cpp
)

# Conditionally exclude CRI-only kernels when DPCPP_SYCL_TARGET is not "cri"
if(NOT DPCPP_SYCL_TARGET STREQUAL "cri")
  foreach(kernel ${CRI_ONLY_KERNELS})
    list(FILTER device_cpp EXCLUDE REGEX ".*${kernel}$")
  endforeach()
  message(STATUS "Excluding CRI kernels (DPCPP_SYCL_TARGET=${DPCPP_SYCL_TARGET}): ${CRI_ONLY_KERNELS}")
else()
  message(STATUS "Including CRI kernels (DPCPP_SYCL_TARGET=${DPCPP_SYCL_TARGET}): ${CRI_ONLY_KERNELS}")
endif()

list(APPEND ATen_XPU_CPP_SRCS ${host_cpp})
list(APPEND ATen_XPU_SYCL_SRCS ${device_cpp})

set(ATen_XPU_CPP_SRCS ${ATen_XPU_CPP_SRCS} PARENT_SCOPE)
set(ATen_XPU_SYCL_SRCS ${ATen_XPU_SYCL_SRCS} PARENT_SCOPE)

include(BuildOnLinux.cmake)
