// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <sycl/ext/oneapi/experimental/clock.hpp>
#include <sycl/sycl.hpp>

namespace sgl_kernel::stage::xe20 {

template <int Groups, int Local, int Rows>
struct Launch {
  static constexpr int work_groups = Groups;
  static constexpr int work_group_size = Local;
  static constexpr int max_rows = Rows;
};

using Flag = sycl::atomic_ref<
    uint32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::device,
    sycl::access::address_space::global_space>;

inline uint64_t clock_now() {
  return sycl::ext::oneapi::experimental::clock<sycl::ext::oneapi::experimental::clock_scope::sub_group>();
}

// SLM-only synchronization preserves unrelated global preloads.
inline void wg_bar(sycl::group<1> group) {
#ifdef __SYCL_DEVICE_ONLY__
  __spirv_ControlBarrier(
      __spv::Scope::Workgroup,
      __spv::Scope::Workgroup,
      __spv::MemorySemanticsMask::AcquireRelease | __spv::MemorySemanticsMask::WorkgroupMemory);
#else
  sycl::group_barrier(group);
#endif
}

}  // namespace sgl_kernel::stage::xe20
