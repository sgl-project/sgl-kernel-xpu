#pragma once

#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

#include "../SYCLHelpers.h"
#include "../Utils.h"

namespace sgl::sampling {

template <typename DType, uint32_t VEC_SIZE>
inline vec_t<DType, VEC_SIZE> load_vec_padded(const DType* row, uint32_t col_base, uint32_t d) {
  vec_t<DType, VEC_SIZE> v(static_cast<DType>(0));
  if (col_base < d) {
    v.load(0, sycl::multi_ptr<const DType, sycl::access::address_space::global_space>(row + col_base));
  }
  return v;
}

template <typename DType, uint32_t VEC_SIZE, uint32_t kWgSize, typename Group>
inline float get_max_value(const Group& grp, const DType* probs, uint32_t row_idx, uint32_t tx, uint32_t d) {
  const DType* row = probs + static_cast<size_t>(row_idx) * static_cast<size_t>(d);
  float thread_max = -std::numeric_limits<float>::infinity();
  for (uint32_t i = 0; i < div_up(d, kWgSize * VEC_SIZE); ++i) {
    const uint32_t col_base = (i * kWgSize + tx) * VEC_SIZE;
    const auto v = load_vec_padded<DType, VEC_SIZE>(row, col_base, d);
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      thread_max = sycl::max(thread_max, static_cast<float>(v[j]));
    }
  }
  return sycl::reduce_over_group(grp, thread_max, sycl::maximum<float>());
}

template <typename DType, uint32_t VEC_SIZE, uint32_t kWgSize, typename Item, typename Predicate>
inline void device_sampling_from_prob(
    const Item& item,
    uint32_t i,
    uint32_t d,
    Predicate pred,
    float u,
    const vec_t<DType, VEC_SIZE>& probs_vec,
    float& aggregate,
    const sycl::local_accessor<int32_t, 1>& sampled_id,
    const sycl::local_accessor<int32_t, 1>& last_valid_id) {
  auto grp = item.get_group();
  const uint32_t tx = item.get_local_id(0);
  const uint32_t col_base = (i * kWgSize + tx) * VEC_SIZE;

  bool valid[VEC_SIZE];
  float local_prefix[VEC_SIZE];
  float running = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    const bool inb = col_base + j < d;
    const float x = inb ? static_cast<float>(probs_vec[j]) : 0.0f;
    valid[j] = inb && pred(x);
    running += valid[j] ? x : 0.0f;
    local_prefix[j] = running;
    if (valid[j]) {
      sycl::atomic_ref<
          int32_t,
          sycl::memory_order::relaxed,
          sycl::memory_scope::work_group,
          sycl::access::address_space::local_space>(last_valid_id[0])
          .fetch_max(static_cast<int32_t>(col_base + j));
    }
  }
  const float thread_total = running;

  const float thread_excl = sycl::exclusive_scan_over_group(grp, thread_total, sycl::plus<float>());
  const float block_total = sycl::reduce_over_group(grp, thread_total, sycl::plus<float>());

  if (aggregate + block_total > u) {
    const float cdf_before_thread = aggregate + thread_excl;
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      if (valid[j] && (cdf_before_thread + local_prefix[j] > u)) {
        sycl::atomic_ref<
            int32_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::work_group,
            sycl::access::address_space::local_space>(sampled_id[0])
            .fetch_min(static_cast<int32_t>(col_base + j));
      }
    }
  }

  aggregate += block_total;
  item.barrier(sycl::access::fence_space::local_space);
}

}  // namespace sgl::sampling
