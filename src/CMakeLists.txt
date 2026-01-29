# ATen XPU sources

file(GLOB device_cpp "sycl/*.cpp" "sycl/*.sycl")
file(GLOB host_cpp "./*.cpp" "./*.cc")

# Exclude kernels for alternative SYCL-TLA builds
list(REMOVE_ITEM device_cpp "${CMAKE_CURRENT_SOURCE_DIR}/sycl/dsv3_router_gemm.cpp")

list(APPEND ATen_XPU_CPP_SRCS ${host_cpp})
list(APPEND ATen_XPU_SYCL_SRCS ${device_cpp})

set(ATen_XPU_CPP_SRCS ${ATen_XPU_CPP_SRCS} PARENT_SCOPE)
set(ATen_XPU_SYCL_SRCS ${ATen_XPU_SYCL_SRCS} PARENT_SCOPE)

include(BuildOnLinux.cmake)
