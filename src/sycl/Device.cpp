#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

static inline syclex::architecture get_device_architecture(at::DeviceIndex device_index = -1) {
  auto device_id = (device_index == -1) ? c10::xpu::current_device() : device_index;
  auto raw_device = c10::xpu::get_raw_device(device_id);
  return raw_device.get_info<syclex::info::device::architecture>();
}

std::tuple<int64_t, int64_t> query_device(int64_t device_index = -1) {
  auto device_arch = get_device_architecture(device_index);
  switch (device_arch) {
    case syclex::architecture::intel_gpu_bmg_g21:
      return std::make_tuple(2, 0);
    // more arch is coming soon
    default:
      throw std::runtime_error("Unsupported XPU architecture.");
  }
}
