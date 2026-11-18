// Adapted from vllm-xpu-kernels csrc/dsa_topk (5bc6a53a).
#pragma once

// SYCL subgroup primitives for DSA top-k. A logical warp is a 32-wide sub_group.

#include <cstdint>
#include <sycl/sycl.hpp>

namespace sgl_kernel {
namespace dsa_topk {

inline constexpr uint32_t kWarpSize = 32;

// Relaxed work-group atomics on shared local memory.
using SlmAtomic = sycl::atomic_ref<
    uint32_t,
    sycl::memory_order::relaxed,
    sycl::memory_scope::work_group,
    sycl::access::address_space::local_space>;

inline uint32_t slm_atomic_add(uint32_t& addr, uint32_t v) {
  return SlmAtomic(addr).fetch_add(v);
}
inline uint32_t slm_atomic_sub(uint32_t& addr, uint32_t v) {
  return SlmAtomic(addr).fetch_sub(v);
}
inline uint32_t slm_atomic_or(uint32_t& addr, uint32_t v) {
  return SlmAtomic(addr).fetch_or(v);
}

// Broadcast lane `src` to the whole subgroup.
template <typename T>
inline T warp_broadcast(const sycl::sub_group& sg, T value, uint32_t src = 0) {
  return sycl::select_from_group(sg, value, src);
}

// Inclusive prefix sum within the subgroup.
template <typename T>
inline T warp_inclusive_sum(const sycl::sub_group& sg, T val) {
  return sycl::inclusive_scan_over_group(sg, val, sycl::plus<T>());
}

// Sum reduction across the subgroup; every lane gets the result.
template <typename T>
inline T warp_reduce_sum(const sycl::sub_group& sg, T val) {
  return sycl::reduce_over_group(sg, val, sycl::plus<T>());
}

// Count true predicates across all lanes in the subgroup.
inline uint32_t warp_sum_bool(const sycl::sub_group& sg, bool pred) {
  return sycl::reduce_over_group(sg, static_cast<uint32_t>(pred ? 1u : 0u), sycl::plus<uint32_t>());
}

// Partial variant: count true predicates only among lanes [0, active).
inline uint32_t warp_sum_bool_masked(const sycl::sub_group& sg, bool pred, uint32_t active) {
  const uint32_t lane = sg.get_local_linear_id();
  const uint32_t v = (lane < active && pred) ? 1u : 0u;
  return sycl::reduce_over_group(sg, v, sycl::plus<uint32_t>());
}

// Each matching lane claims one slot; other lanes return UINT32_MAX.
// One atomic per subgroup reduces contention; slot order is subgroup-local.
inline uint32_t slm_atomic_add_agg(const sycl::sub_group& sg, uint32_t& counter, bool pred) {
  const uint32_t inc = pred ? 1u : 0u;
  const uint32_t rank = sycl::exclusive_scan_over_group(sg, inc, sycl::plus<uint32_t>());
  const uint32_t total = sycl::reduce_over_group(sg, inc, sycl::plus<uint32_t>());
  uint32_t base = 0;
  if (sg.leader() && total != 0) base = SlmAtomic(counter).fetch_add(total);
  base = sycl::group_broadcast(sg, base, 0);
  return pred ? base + rank : 0xFFFFFFFFu;
}

// Order-preserving float -> uint32 key (larger float -> larger key).
inline uint32_t extract_exact_bin(float x) {
  const uint32_t bits = sycl::bit_cast<uint32_t>(x);
  if ((bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0) return 0xFFFFFFFFu;
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

}  // namespace dsa_topk
}  // namespace sgl_kernel
