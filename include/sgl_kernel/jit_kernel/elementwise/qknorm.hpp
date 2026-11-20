/**
 * QKNorm SYCL Kernel for SGLang
 *
 * Fused in-place QK normalization for Intel XPU.
 * Applies RMSNorm independently to each Q and K head:
 *   q[i] = q[i] / sqrt(mean(q[i]^2) + eps) * q_weight
 *   k[i] = k[i] / sqrt(mean(k[i]^2) + eps) * k_weight
 *
 * Layout:
 *   q: [num_tokens, num_qo_heads, head_dim]
 *   k: [num_tokens, num_kv_heads, head_dim]
 *   q_weight, k_weight: [head_dim]
 *
 * Optimizations (mirrors the warp path of src/sycl/FusedQKNormRope.cpp):
 * - Warp path: one sub-group per (token, head) row; each lane owns
 *   head_dim / 16 contiguous elements kept in registers, and the sum of
 *   squares is reduced within the sub-group only (no SLM / work-group barrier)
 * - Vector width picked at launch from the device's preferred width, pointer
 *   alignment and token strides (falls back to narrower widths, never fails)
 * - Occupancy-aware persistent launch with a grid-stride loop over rows
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <sycl/sycl.hpp>
#include <unordered_map>

#include "../memory.hpp"

namespace sgl {
namespace sycl_kernel {

// Import aligned_vector from memory utilities
using sgl::sycl::aligned_vector;

static constexpr int kQKNormSubGroupSize = 16;  // same setting as NUM_REDUCE_STAGES in FusedQKNormRope.cpp

// ============================================================================
// Device / launch helpers
// ============================================================================

struct QKNormDeviceInfo {
  int64_t max_wg_size;
  int64_t pref_vec_width_short;     // preferred vector width for 2-byte elements
  int64_t max_resident_work_items;  // 0 if unknown
};

inline QKNormDeviceInfo qknorm_device_info(const ::sycl::device& dev) {
  static std::mutex mutex;
  static std::unordered_map<::sycl::device, QKNormDeviceInfo> cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = cache.find(dev);
  if (it != cache.end()) return it->second;

  QKNormDeviceInfo info{};
  info.max_wg_size = static_cast<int64_t>(dev.get_info<::sycl::info::device::max_work_group_size>());
  info.pref_vec_width_short = std::max<int64_t>(1, dev.get_info<::sycl::info::device::preferred_vector_width_short>());
  // Resident work-item capacity, expressed in this kernel's sub-group width
  // (same as FusedQKNormRope.cpp: EUs * HW threads per EU * sub-group size).
  info.max_resident_work_items = 0;
  if (dev.has(::sycl::aspect::ext_intel_gpu_eu_count) && dev.has(::sycl::aspect::ext_intel_gpu_hw_threads_per_eu)) {
    const int64_t eu_count = dev.get_info<::sycl::ext::intel::info::device::gpu_eu_count>();
    const int64_t hw_threads = dev.get_info<::sycl::ext::intel::info::device::gpu_hw_threads_per_eu>();
    info.max_resident_work_items = eu_count * hw_threads * kQKNormSubGroupSize;
  }
  cache.emplace(dev, info);
  return info;
}

// Largest power-of-two width (<= max_vec_size) that divides elems_per_thread
// and for which every pointer is aligned to elem_size * vec_size bytes and
// every element stride is a multiple of vec_size. Falls back to 1.
inline int64_t qknorm_pick_aligned_vec_size(
    int64_t max_vec_size,
    int64_t elem_size,
    std::initializer_list<const void*> ptrs,
    int64_t elems_per_thread,
    std::initializer_list<int64_t> elem_strides) {
  for (int64_t vec_size = max_vec_size; vec_size > 1; vec_size >>= 1) {
    if (elems_per_thread % vec_size != 0) continue;
    const int64_t align_bytes = elem_size * vec_size;
    bool ok = true;
    for (const void* p : ptrs) {
      if (reinterpret_cast<uintptr_t>(p) % align_bytes != 0) {
        ok = false;
        break;
      }
    }
    for (int64_t stride : elem_strides) {
      if (stride % vec_size != 0) {
        ok = false;
        break;
      }
    }
    if (ok) return vec_size;
  }
  return 1;
}

// Dispatches to the kernel instantiation for vec_size in {16, 8, 4, 2},
// accepting only widths that divide kElemsPerThread, else scalar (1).
template <int64_t kElemsPerThread, typename Fn>
inline void qknorm_dispatch_vec_size(int64_t vec_size, Fn&& fn) {
  bool dispatched = false;
  auto try_vec_size = [&](auto vec_size_tag) {
    constexpr int64_t kCandidate = decltype(vec_size_tag)::value;
    if constexpr (kCandidate <= kElemsPerThread && kElemsPerThread % kCandidate == 0) {
      if (!dispatched && vec_size == kCandidate) {
        fn(vec_size_tag);
        dispatched = true;
      }
    }
  };
  try_vec_size(std::integral_constant<int64_t, 16>{});
  try_vec_size(std::integral_constant<int64_t, 8>{});
  try_vec_size(std::integral_constant<int64_t, 4>{});
  try_vec_size(std::integral_constant<int64_t, 2>{});
  if (!dispatched) {
    fn(std::integral_constant<int64_t, 1>{});
  }
}

struct QKNormLaunchConfig {
  int64_t block_size;
  int64_t grid_size;
};

// Persistent launch: sub-groups per work-group capped by the device max work-group
// size (and 512 for 16-wide sub-groups), grid capped by resident work-item capacity.
inline QKNormLaunchConfig qknorm_launch_config(const QKNormDeviceInfo& info, int64_t total_work) {
  int64_t max_wg_size = std::min<int64_t>(info.max_wg_size, 512);
  max_wg_size = std::max<int64_t>(max_wg_size, kQKNormSubGroupSize);

  const int64_t max_subgroups_per_wg = std::max<int64_t>(1, max_wg_size / kQKNormSubGroupSize);
  const int64_t subgroups_per_wg = std::max<int64_t>(1, std::min<int64_t>(total_work, max_subgroups_per_wg));
  const int64_t block_size = subgroups_per_wg * kQKNormSubGroupSize;

  const int64_t needed_blocks = (total_work + subgroups_per_wg - 1) / subgroups_per_wg;
  int64_t grid_size = needed_blocks;
  if (info.max_resident_work_items > 0) {
    const int64_t max_resident_blocks = std::max<int64_t>(1, info.max_resident_work_items / block_size);
    grid_size = std::min(grid_size, max_resident_blocks);
  }
  return {block_size, std::max<int64_t>(1, grid_size)};
}

// ============================================================================
// QKNorm Kernel — warp path: one sub-group per (token, head) row
// ============================================================================

template <typename T, int64_t kHeadDim, int64_t kVecSize>
class QKNormKernel {
 public:
  static_assert(kHeadDim % kQKNormSubGroupSize == 0, "head_dim must be divisible by the sub-group size");
  static constexpr int64_t kElemsPerThread = kHeadDim / kQKNormSubGroupSize;
  static_assert(kElemsPerThread % kVecSize == 0, "kVecSize must evenly divide kElemsPerThread");
  static constexpr int64_t kNumChunks = kElemsPerThread / kVecSize;
  using Vec = aligned_vector<T, kVecSize>;

  QKNormKernel(
      T* q,
      T* k,
      const T* q_weight,
      const T* k_weight,
      int64_t q_stride,
      int64_t k_stride,
      uint32_t num_qo_heads,
      uint32_t num_kv_heads,
      uint32_t num_tokens,
      float eps)
      : q_(q),
        k_(k),
        q_weight_(q_weight),
        k_weight_(k_weight),
        q_stride_(q_stride),
        k_stride_(k_stride),
        num_qo_heads_(num_qo_heads),
        num_kv_heads_(num_kv_heads),
        num_tokens_(num_tokens),
        eps_(eps) {}

  [[sycl::reqd_sub_group_size(kQKNormSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
    const auto sg = item.get_sub_group();
    const uint32_t lane_id = static_cast<uint32_t>(item.get_local_id(0) % kQKNormSubGroupSize);
    const uint32_t warp_id = static_cast<uint32_t>(item.get_local_id(0) / kQKNormSubGroupSize);
    const uint32_t warps_per_block = static_cast<uint32_t>(item.get_local_range(0) / kQKNormSubGroupSize);
    const uint32_t start_worker_id = static_cast<uint32_t>(item.get_group(0)) * warps_per_block + warp_id;
    const uint32_t num_workers = static_cast<uint32_t>(item.get_group_range(0)) * warps_per_block;

    const uint32_t num_q_and_k = num_qo_heads_ + num_kv_heads_;
    const uint32_t num_works = num_q_and_k * num_tokens_;

    // Grid-stride loop: the launch is persistent, so a sub-group may own several rows.
    for (uint32_t idx = start_worker_id; idx < num_works; idx += num_workers) {
      const uint32_t token_id = idx / num_q_and_k;
      const uint32_t head_id = idx % num_q_and_k;
      const bool is_q = head_id < num_qo_heads_;

      T* row_ptr;
      const T* weight_ptr;
      if (is_q) {
        row_ptr = q_ + token_id * q_stride_ + static_cast<int64_t>(head_id) * kHeadDim;
        weight_ptr = q_weight_;
      } else {
        const uint32_t k_head_id = head_id - num_qo_heads_;
        row_ptr = k_ + token_id * k_stride_ + static_cast<int64_t>(k_head_id) * kHeadDim;
        weight_ptr = k_weight_;
      }

      const int64_t lane_offset = static_cast<int64_t>(lane_id) * kElemsPerThread;
      float elems[kElemsPerThread];
      float sum_of_squares = 0.0f;
#pragma unroll
      for (int64_t c = 0; c < kNumChunks; ++c) {
        const Vec in_vec = *reinterpret_cast<const Vec*>(row_ptr + lane_offset + c * kVecSize);
#pragma unroll
        for (int64_t v = 0; v < kVecSize; ++v) {
          const float x = static_cast<float>(in_vec[v]);
          elems[c * kVecSize + v] = x;
          sum_of_squares += x * x;
        }
      }

      sum_of_squares = ::sycl::reduce_over_group(sg, sum_of_squares, ::sycl::plus<float>());
      const float norm_factor = ::sycl::rsqrt(sum_of_squares / static_cast<float>(kHeadDim) + eps_);

#pragma unroll
      for (int64_t c = 0; c < kNumChunks; ++c) {
        const Vec w_vec = *reinterpret_cast<const Vec*>(weight_ptr + lane_offset + c * kVecSize);
        Vec out_vec;
#pragma unroll
        for (int64_t v = 0; v < kVecSize; ++v) {
          out_vec[v] = static_cast<T>(elems[c * kVecSize + v] * norm_factor * static_cast<float>(w_vec[v]));
        }
        *reinterpret_cast<Vec*>(row_ptr + lane_offset + c * kVecSize) = out_vec;
      }
    }
  }

 private:
  T* q_;
  T* k_;
  const T* q_weight_;
  const T* k_weight_;
  int64_t q_stride_;
  int64_t k_stride_;
  uint32_t num_qo_heads_;
  uint32_t num_kv_heads_;
  uint32_t num_tokens_;
  float eps_;
};

// ============================================================================
// Launcher
// ============================================================================

template <int64_t kHeadDim, typename DType>
void qknorm_launcher(
    ::sycl::queue& queue,
    void* q,
    void* k,
    const void* q_weight,
    const void* k_weight,
    int64_t q_stride,
    int64_t k_stride,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t num_tokens,
    float eps) {
  constexpr int64_t kElemsPerThread = kHeadDim / kQKNormSubGroupSize;
  const int64_t total_work = static_cast<int64_t>(num_qo_heads + num_kv_heads) * num_tokens;
  if (total_work == 0) return;

  DType* q_ptr = static_cast<DType*>(q);
  DType* k_ptr = static_cast<DType*>(k);
  const DType* qw_ptr = static_cast<const DType*>(q_weight);
  const DType* kw_ptr = static_cast<const DType*>(k_weight);

  const QKNormDeviceInfo info = qknorm_device_info(queue.get_device());
  const QKNormLaunchConfig cfg = qknorm_launch_config(info, total_work);

  // q/k may be slices of a packed QKV buffer, so the vector width must also
  // divide the token strides, not just satisfy base-pointer alignment.
  static_assert(sizeof(DType) == 2, "QKNorm supports 2-byte dtypes (fp16/bf16) only");
  const int64_t max_vec_size = std::min<int64_t>(kElemsPerThread, info.pref_vec_width_short);
  const int64_t vec_size = qknorm_pick_aligned_vec_size(
      max_vec_size, sizeof(DType), {q_ptr, k_ptr, qw_ptr, kw_ptr}, kElemsPerThread, {q_stride, k_stride});

  qknorm_dispatch_vec_size<kElemsPerThread>(vec_size, [&](auto vec_size_tag) {
    constexpr int64_t kVecSize = decltype(vec_size_tag)::value;
    queue.parallel_for(
        ::sycl::nd_range<1>(
            ::sycl::range<1>(static_cast<size_t>(cfg.grid_size * cfg.block_size)),
            ::sycl::range<1>(static_cast<size_t>(cfg.block_size))),
        QKNormKernel<DType, kHeadDim, kVecSize>(
            q_ptr, k_ptr, qw_ptr, kw_ptr, q_stride, k_stride, num_qo_heads, num_kv_heads, num_tokens, eps));
  });
}

// ============================================================================
// C API for Python Binding
// ============================================================================

// Two-level macro to ensure macro arguments are expanded before token pasting.
#define _DEFINE_QKNORM_FORWARD(DTYPE_SUFFIX, DTYPE, HEAD_DIM)                                              \
  extern "C" void qknorm_forward_##DTYPE_SUFFIX##_##HEAD_DIM(                                              \
      void* queue_ptr,                                                                                     \
      void* q,                                                                                             \
      void* k,                                                                                             \
      const void* q_weight,                                                                                \
      const void* k_weight,                                                                                \
      int64_t q_stride,                                                                                    \
      int64_t k_stride,                                                                                    \
      uint32_t num_qo_heads,                                                                               \
      uint32_t num_kv_heads,                                                                               \
      uint32_t num_tokens,                                                                                 \
      float eps) {                                                                                         \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr);                                                 \
    qknorm_launcher<HEAD_DIM, DTYPE>(                                                                      \
        queue, q, k, q_weight, k_weight, q_stride, k_stride, num_qo_heads, num_kv_heads, num_tokens, eps); \
  }
#define DEFINE_QKNORM_FORWARD(DTYPE_SUFFIX, DTYPE, HEAD_DIM) _DEFINE_QKNORM_FORWARD(DTYPE_SUFFIX, DTYPE, HEAD_DIM)

#define DEFINE_QKNORM_ALL_DTYPES(HEAD_DIM)            \
  DEFINE_QKNORM_FORWARD(fp16, ::sycl::half, HEAD_DIM) \
  DEFINE_QKNORM_FORWARD(bf16, ::sycl::ext::oneapi::bfloat16, HEAD_DIM)

// When SGL_QKNORM_HEAD_DIM is defined, compile only the requested variant.
#ifdef SGL_QKNORM_HEAD_DIM
#if defined(SGL_QKNORM_DTYPE_fp16)
DEFINE_QKNORM_FORWARD(fp16, ::sycl::half, SGL_QKNORM_HEAD_DIM)
#elif defined(SGL_QKNORM_DTYPE_bf16)
DEFINE_QKNORM_FORWARD(bf16, ::sycl::ext::oneapi::bfloat16, SGL_QKNORM_HEAD_DIM)
#else
DEFINE_QKNORM_ALL_DTYPES(SGL_QKNORM_HEAD_DIM)
#endif
#else
DEFINE_QKNORM_ALL_DTYPES(64)
DEFINE_QKNORM_ALL_DTYPES(128)
DEFINE_QKNORM_ALL_DTYPES(256)
DEFINE_QKNORM_ALL_DTYPES(512)
DEFINE_QKNORM_ALL_DTYPES(1024)
#endif

#undef DEFINE_QKNORM_FORWARD
#undef _DEFINE_QKNORM_FORWARD
#undef DEFINE_QKNORM_ALL_DTYPES

}  // namespace sycl_kernel
}  // namespace sgl
