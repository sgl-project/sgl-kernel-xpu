#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

#include "../SYCLHelpers.h"
#include "../Utils.h"

namespace sgl::sampling {

template <typename DType, uint32_t VEC_SIZE, uint32_t kWgSize, typename Group>
inline float get_max_value(const Group& grp, const DType* probs, uint32_t row_idx, uint32_t tx, uint32_t d) {
  const size_t row_offset = static_cast<size_t>(row_idx) * static_cast<size_t>(d);
  float thread_max = 0.0f;
  for (uint32_t i = 0; i < div_up(d, kWgSize * VEC_SIZE); ++i) {
    vec_t<DType, VEC_SIZE> v(static_cast<DType>(0));
    if ((i * kWgSize + tx) * VEC_SIZE < d) {
      v.load(
          0,
          sycl::multi_ptr<const DType, sycl::access::address_space::global_space>(
              probs + row_offset + (i * kWgSize + tx) * VEC_SIZE));
    }
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      thread_max = sycl::max(thread_max, static_cast<float>(v[j]));
    }
  }
  return sycl::reduce_over_group(grp, thread_max, sycl::maximum<float>());
}

// Slower than `sycl::inclusive_scan_over_group` but its deterministic.
template <uint32_t VEC_SIZE, uint32_t kWgSize, typename Item>
inline void deterministic_inclusive_sum(
    const Item& item,
    const float (&in_data)[VEC_SIZE],
    float (&out_data)[VEC_SIZE],
    const sycl::local_accessor<float, 1>& smem_prefix_sum) {
  constexpr uint32_t WARP_SIZE = 32;
  constexpr uint32_t NUM_WARPS = kWgSize / WARP_SIZE;
  auto sg = item.get_sub_group();
  const uint32_t tx = item.get_local_id(0);

  float thread_data[VEC_SIZE];
  float thread_sum = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    thread_sum += in_data[j];
    thread_data[j] = thread_sum;
  }
  float thread_exclusive_prefix_sum = thread_sum;

#pragma unroll
  for (uint32_t offset = 1; offset < WARP_SIZE; offset *= 2) {
    const float tmp = sycl::shift_group_right(sg, thread_exclusive_prefix_sum, offset);
    if ((tx + 1) % (offset * 2) == 0) {
      thread_exclusive_prefix_sum += tmp;
    }
  }
  const float warp_sum = sycl::select_from_group(sg, thread_exclusive_prefix_sum, WARP_SIZE - 1);
  if (tx % WARP_SIZE == WARP_SIZE - 1) {
    thread_exclusive_prefix_sum = 0.0f;
  }
#pragma unroll
  for (uint32_t offset = WARP_SIZE / 2; offset >= 1; offset /= 2) {
    const float tmp = sycl::permute_group_by_xor(sg, thread_exclusive_prefix_sum, offset);
    if ((tx + 1) % (offset * 2) == 0) {
      thread_exclusive_prefix_sum += tmp;
    } else if ((tx + 1) % (offset * 2) == offset) {
      thread_exclusive_prefix_sum = tmp;
    }
  }

  smem_prefix_sum[tx / WARP_SIZE] = warp_sum;
  item.barrier(sycl::access::fence_space::local_space);

  if (tx < WARP_SIZE) {
    float warp_exclusive_prefix_sum = (tx < NUM_WARPS) ? smem_prefix_sum[tx] : 0.0f;
#pragma unroll
    for (uint32_t offset = 1; offset < WARP_SIZE; offset *= 2) {
      const float tmp = sycl::shift_group_right(sg, warp_exclusive_prefix_sum, offset);
      if ((tx + 1) % (offset * 2) == 0) {
        warp_exclusive_prefix_sum += tmp;
      }
    }
    if (tx % WARP_SIZE == WARP_SIZE - 1) {
      warp_exclusive_prefix_sum = 0.0f;
    }
#pragma unroll
    for (uint32_t offset = WARP_SIZE / 2; offset >= 1; offset /= 2) {
      const float tmp = sycl::permute_group_by_xor(sg, warp_exclusive_prefix_sum, offset);
      if ((tx + 1) % (offset * 2) == 0) {
        warp_exclusive_prefix_sum += tmp;
      } else if ((tx + 1) % (offset * 2) == offset) {
        warp_exclusive_prefix_sum = tmp;
      }
    }
    if (tx < NUM_WARPS) {
      smem_prefix_sum[tx] = warp_exclusive_prefix_sum;
    }
  }
  item.barrier(sycl::access::fence_space::local_space);

#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    out_data[j] = smem_prefix_sum[tx / WARP_SIZE] + thread_exclusive_prefix_sum + thread_data[j];
  }
}

template <typename DType, uint32_t VEC_SIZE, uint32_t kWgSize, bool DETERMINISTIC, typename Item, typename Predicate>
inline void device_sampling_from_prob(
    const Item& item,
    uint32_t i,
    uint32_t d,
    Predicate pred,
    float u,
    const vec_t<DType, VEC_SIZE>& probs_vec,
    float& aggregate,
    const sycl::local_accessor<int32_t, 1>& sampled_id,
    const sycl::local_accessor<int32_t, 1>& last_valid_id,
    const sycl::local_accessor<float, 1>& smem_prefix_sum) {
  auto grp = item.get_group();
  const uint32_t tx = item.get_local_id(0);
  const uint32_t col_base = (i * kWgSize + tx) * VEC_SIZE;

  bool valid[VEC_SIZE];
  float prob_greater_than_threshold[VEC_SIZE];
  float local_prefix[VEC_SIZE];
  float running = 0.0f;
#pragma unroll
  for (uint32_t j = 0; j < VEC_SIZE; ++j) {
    const bool inb = col_base + j < d;
    const float x = inb ? static_cast<float>(probs_vec[j]) : 0.0f;
    valid[j] = inb && pred(x);
    prob_greater_than_threshold[j] = valid[j] ? x : 0.0f;
    running += prob_greater_than_threshold[j];
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

  const float aggregate_local = sycl::reduce_over_group(grp, thread_total, sycl::plus<float>());

  if (aggregate + aggregate_local > u) {
    float inclusive_cdf[VEC_SIZE];
    if constexpr (DETERMINISTIC) {
      deterministic_inclusive_sum<VEC_SIZE, kWgSize>(item, prob_greater_than_threshold, inclusive_cdf, smem_prefix_sum);
    } else {
      const float thread_excl = sycl::exclusive_scan_over_group(grp, thread_total, sycl::plus<float>());
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        inclusive_cdf[j] = thread_excl + local_prefix[j];
      }
    }

#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      if (valid[j] && (aggregate + inclusive_cdf[j] > u)) {
        sycl::atomic_ref<
            int32_t,
            sycl::memory_order::relaxed,
            sycl::memory_scope::work_group,
            sycl::access::address_space::local_space>(sampled_id[0])
            .fetch_min(static_cast<int32_t>(col_base + j));
      }
    }
  }

  aggregate += aggregate_local;
  item.barrier(sycl::access::fence_space::local_space);
}

}  // namespace sgl::sampling
