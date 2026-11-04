#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "sgl_kernel_export.h"

// Compiled twice: into each backend's device library (for the `query_device`
// op) and into the standalone libsgl_arch_probe.so, which python/sgl_kernel/_arch.py
// loads with ctypes to pick the backend before any backend is imported. Keep this
// file free of operator registrations and kernel launches so both stay valid.

namespace syclex = sycl::ext::oneapi::experimental;

static inline syclex::architecture get_device_architecture(at::DeviceIndex device_index = -1) {
  auto device_id = (device_index == -1) ? c10::xpu::current_device() : device_index;
  auto raw_device = c10::xpu::get_raw_device(device_id);
  return raw_device.get_info<syclex::info::device::architecture>();
}

SGL_KERNEL_EXPORT std::tuple<int64_t, int64_t> query_device(int64_t device_index = -1) {
  auto device_arch = get_device_architecture(device_index);
  switch (device_arch) {
    case syclex::architecture::intel_gpu_bmg_g21:
    case syclex::architecture::intel_gpu_bmg_g31:
      return std::make_tuple(2, 0);
// Need oneAPI 2026.2 or later to support `architecture::intel_gpu_cri`. Check the release
// (__INTEL_LLVM_COMPILER) rather than the build date (__SYCL_COMPILER_VERSION): 2026.1.x
// patch releases (e.g. 2026.1.3, 20260724) are built later but still lack this enum.
// The host compiler (g++) uses the release number extracted by CMake from icpx.
#if __INTEL_LLVM_COMPILER >= 20260200 || \
    (!defined(__INTEL_LLVM_COMPILER) && SGL_INTEL_LLVM_COMPILER_VERSION >= 20260200)
    case syclex::architecture::intel_gpu_cri:
      return std::make_tuple(3, 5);
#endif
    // more arch is coming soon
    default:
      throw std::runtime_error("Unsupported XPU architecture.");
  }
}

// C entry point for the architecture probe. Out-parameters rather than a returned
// std::tuple, which comes back through a hidden sret pointer and is not safely
// callable through ctypes. Returns 0 on success, -1 for null outputs, -2 if the
// device query fails or the architecture is unsupported.
extern "C" SGL_KERNEL_EXPORT int sgl_probe_device_capability(int64_t device_index, int64_t* major, int64_t* minor) {
  if (major == nullptr || minor == nullptr) {
    return -1;
  }
  try {
    auto [maj, min] = query_device(device_index);
    *major = maj;
    *minor = min;
    return 0;
  } catch (...) {
    return -2;
  }
}
