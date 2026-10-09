#include <c10/xpu/XPUFunctions.h>
#include <c10/xpu/XPUStream.h>

#include <sycl/sycl.hpp>

#include "sgl_arch_query.h"
#include "sgl_kernel_export.h"

SGL_KERNEL_EXPORT std::tuple<int64_t, int64_t> query_device(int64_t device_index = -1) {
  return sgl::device_capability(device_index);
}
