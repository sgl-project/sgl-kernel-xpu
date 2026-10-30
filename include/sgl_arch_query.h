/***************************************************************************************************
 * Copyright 2026 Intel corporation. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 **************************************************************************************************/
#pragma once

#include <c10/xpu/XPUFunctions.h>

#include <cstdint>
#include <sycl/sycl.hpp>
#include <tuple>

// Single source of truth for the XPU architecture -> (major, minor) capability
// mapping. Shared by the `query_device` torch op (src/sycl/Device.cpp) and by
// the standalone probe (src/sgl_arch_probe.cpp) that selects the per-arch
// backend before the main extension is imported.
namespace sgl {

namespace syclex = sycl::ext::oneapi::experimental;

inline syclex::architecture device_architecture(int64_t device_index = -1) {
  auto device_id = (device_index == -1) ? c10::xpu::current_device() : static_cast<int>(device_index);
  return c10::xpu::get_raw_device(device_id).get_info<syclex::info::device::architecture>();
}

inline std::tuple<int64_t, int64_t> device_capability(int64_t device_index = -1) {
  switch (device_architecture(device_index)) {
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

}  // namespace sgl
