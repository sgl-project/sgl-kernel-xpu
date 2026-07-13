#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <iostream>
#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

#include "MemoryAccess.h"
#include "Norm.h"
#include "SYCLHelpers.h"
#include "Utils.h"
#include "cutlass/float8.h"

// TODO: Remove this when sycl float8 is supported
using cutlass::float_e4m3_t;

namespace at::native::xpu {

template <typename T>
inline T divUp(T m, T n) {
  return (m + n - 1) / n;
}

// Sub-group reduction for sum; result is broadcast to every lane.
template <typename T>
inline T subGroupReduceSum(T val, const sycl::sub_group& sg) {
  return sycl::reduce_over_group(sg, val, sycl::plus<T>());
}

inline float compute_freq_yarn(float base, int rotary_dim, int half_dim, float factor, float low, float high) {
  // freq_idx is the value from arange(0, rotary_dim, 2): i.e., 0, 2, 4, 6, ...
  float freq = sycl::pow(base, -2.0f * static_cast<float>(half_dim) / static_cast<float>(rotary_dim));

  if (factor != 1.0f) {
    float inv_freq_extrapolation = freq;
    float inv_freq_interpolation = freq / factor;

    float high_adj = high;
    if (sycl::fabs(low - high_adj) <= 1e-6f) {
      high_adj += 0.001f;
    }

    // Match Python: dim_range is [0, 2, 4, 6, ...], so use 2*half_dim
    float dim_value = 2.0f * static_cast<float>(half_dim);
    float linear_func = (dim_value - low) / (high_adj - low);
    float ramp_func = sycl::fmin(sycl::fmax(linear_func, 0.0f), 1.0f);

    // Match Python formula exactly
    freq = inv_freq_interpolation * (1.0f - ramp_func) + inv_freq_extrapolation * ramp_func;
  }

  return freq;
}

// ---------------------------------------------------------------------------
// Sub-group width portability and occupancy-aware (persistent) launch sizing.
//
// Unlike CUDA's fixed 32-wide warp, Intel GPUs support a device-dependent set
// of sub-group widths (commonly a subset of {8, 16, 32}). The kernels below
// pin the width to NUM_REDUCE_STAGES via [[sycl::reqd_sub_group_size(...)]]
// so compile-time lane arithmetic matches the runtime sub-group size, and
// validate at launch time that the device actually supports that width.
inline void check_subgroup_size_supported(int64_t required_size) {
  auto* dev_prop = at::xpu::getDeviceProperties(dpcppGetDeviceIdOfCurrentQueue());
  bool supported = false;
  for (auto sz : dev_prop->sub_group_sizes) {
    if (static_cast<int64_t>(sz) == required_size) {
      supported = true;
      break;
    }
  }
  TORCH_CHECK(
      supported,
      "fused_qk_norm_rope kernels require sub-group size ",
      required_size,
      ", which is not supported by this device");
}

struct PersistentLaunchConfig {
  int64_t blockSize;
  int64_t gridSize;
};

template <typename T>
struct TypeTag {
  using type = T;
};

template <bool kAllowFloat8, typename Fn>
inline void dispatchFusedQKNormRopeScalarType(at::ScalarType scalar_type, const char* kernel_name, Fn&& fn) {
  switch (scalar_type) {
    case at::ScalarType::Half:
      fn(TypeTag<sycl::half>{});
      break;
    case at::ScalarType::BFloat16:
      fn(TypeTag<sycl::ext::oneapi::bfloat16>{});
      break;
    case at::ScalarType::Float8_e4m3fn:
      if constexpr (kAllowFloat8) {
        fn(TypeTag<float_e4m3_t>{});
        break;
      }
      [[fallthrough]];
    default:
      TORCH_CHECK(false, "Unsupported dtype for ", kernel_name, ": ", scalar_type);
  }
}

template <typename Fn>
inline void dispatchFusedQKNormRopePositionsType(at::ScalarType scalar_type, const char* kernel_name, Fn&& fn) {
  switch (scalar_type) {
    case at::ScalarType::Int:
      fn(TypeTag<int32_t>{});
      break;
    case at::ScalarType::Long:
      fn(TypeTag<int64_t>{});
      break;
    default:
      TORCH_CHECK(false, "Unsupported dtype for ", kernel_name, " positions: ", scalar_type);
  }
}

template <typename Fn>
inline void dispatchFusedQKNormRopeHeadDim(int64_t head_dim, const char* kernel_name, Fn&& fn) {
  switch (head_dim) {
    case 64:
      fn(std::integral_constant<int64_t, 64>{});
      break;
    case 128:
      fn(std::integral_constant<int64_t, 128>{});
      break;
    case 256:
      fn(std::integral_constant<int64_t, 256>{});
      break;
    default:
      TORCH_CHECK(false, "Unsupported head dimension for ", kernel_name, ": ", head_dim);
  }
}

// Dispatches over vectorized VecSize instantiations (16/8/4/2), gated by
// `kCandidate <= kElemsPerThread` so only widths that evenly divide the
// per-lane element count are tried, then unconditionally falls back to
// scalar (VecSize=1) -- always valid regardless of alignment -- if none
// matched, guaranteeing the kernel is always launched. Shared by both the
// packed-QKV and cos/sin-cache launch paths.
template <int64_t kElemsPerThread, typename Fn>
inline void dispatchFusedQKNormRopeVecSize(int64_t vec_size, Fn&& fn) {
  bool dispatched = false;
  auto try_vec_size = [&](auto vec_size_tag) {
    constexpr int64_t kCandidate = decltype(vec_size_tag)::value;
    if constexpr (kCandidate <= kElemsPerThread) {
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

// Picks the largest power-of-two width (<= max_vec_size) for which every
// pointer in `ptrs` is aligned to `elem_size * vec_size` bytes, shrinking to
// 1 (always safe) if needed. Only the physical load/store chunk width
// shrinks; the per-lane algorithmic element count is unaffected.
inline int64_t pick_aligned_vec_size(int64_t max_vec_size, int64_t elem_size, std::initializer_list<const void*> ptrs) {
  int64_t vec_size = max_vec_size;
  while (vec_size > 1) {
    const int64_t align_bytes = elem_size * vec_size;
    bool all_aligned = true;
    for (const void* p : ptrs) {
      if (reinterpret_cast<uintptr_t>(p) % align_bytes != 0) {
        all_aligned = false;
        break;
      }
    }
    if (all_aligned) break;
    vec_size >>= 1;
  }
  return vec_size;
}

// Caps the vector load/store chunk at 16 bytes (128-bit), matching
// RMSNorm.cpp's convention, to avoid oversized vector instructions and
// register pressure/spills for wide per-lane element counts.
inline int64_t maxHwVecSize(int64_t elem_size) {
  constexpr int64_t kMaxVecBytes = sizeof(float) * 4;
  return std::max<int64_t>(1, kMaxVecBytes / elem_size);
}

// Caps work-group size for warp-per-(token,head) kernels: work-groups wider
// than 512 sub-groups-of-16 add barrier/sync overhead without improving
// occupancy, so the cap only applies at that width.
inline int64_t capWorkgroupSize(int64_t maxWgSize, int64_t subgroupSize) {
  if (subgroupSize == 16) {
    maxWgSize = std::min<int64_t>(maxWgSize, 512);
  }
  return std::max(maxWgSize, subgroupSize);
}

// Computes an occupancy-aware, persistent-kernel launch config for kernels
// where each sub-group handles one unit of work, with a grid-stride loop
// consuming any remainder.
//  - blockSize: sub-groups per work-group, capped by the max work-group size.
//  - gridSize: capped by the device's resident work-item capacity (rescaled
//    to this kernel's sub-group width) -- the SYCL analog of CUDA's "max
//    active blocks per SM * SM count" -- so oversubscribed work is consumed
//    by the in-kernel grid-stride loop instead of launching extra work-groups.
inline PersistentLaunchConfig computePersistentLaunchConfig(int64_t totalWork, int64_t subgroupSize) {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const int64_t maxWgSize = capWorkgroupSize(dpcppMaxWorkGroupSize(dev_id), subgroupSize);

  const int64_t maxSubgroupsPerWg = std::max<int64_t>(1, maxWgSize / subgroupSize);
  const int64_t subgroupsPerWg = std::max<int64_t>(1, std::min<int64_t>(totalWork, maxSubgroupsPerWg));
  const int64_t blockSize = subgroupsPerWg * subgroupSize;

  const int64_t totalResource =
      std::max<int64_t>(subgroupSize, dpcppMaxWorkItemsPerTile(dev_id) / dpcppMaxSubGroupSize(dev_id) * subgroupSize);
  const int64_t maxResidentBlocks = std::max<int64_t>(1, totalResource / blockSize);
  const int64_t neededBlocks = divUp(totalWork, subgroupsPerWg);
  const int64_t gridSize = std::max<int64_t>(1, std::min(neededBlocks, maxResidentBlocks));

  return {blockSize, gridSize};
}

// SYCL Kernel for Fused QK Norm and RoPE (packed QKV layout, legacy path):
// analytic (YaRN-aware) RoPE frequencies computed on the fly, rather than
// read from a cache. Not used by production sglang model code today (which
// uses the cos_sin_cache op below); kept for existing test/benchmark
// coverage. Each sub-group processes one (token, head) pair from Q or K
// (V heads untouched).
template <int head_dim, bool interleave, typename scalar_t, int VecSize>
struct FusedQKNormRopeKernel {
  scalar_t* qkv;
  int num_heads_q;
  int num_heads_k;
  int num_heads_v;
  float eps;
  const scalar_t* q_weight;
  const scalar_t* k_weight;
  float base;
  const int* position_ids;
  int num_tokens;
  float factor;
  float low;
  float high;
  float attention_factor;
  int rotary_dim;

  [[sycl::reqd_sub_group_size(NUM_REDUCE_STAGES)]] void operator()(sycl::nd_item<1> item) const {
    using accscalar_t = float;
    constexpr int numElemsPerThread = head_dim / NUM_REDUCE_STAGES;
    static_assert(numElemsPerThread % VecSize == 0, "VecSize must evenly divide numElemsPerThread");
    constexpr int numChunks = numElemsPerThread / VecSize;
    using VecT = aligned_vector_loop<scalar_t, VecSize>;

    auto sg = item.get_sub_group();
    const int laneId = static_cast<int>(item.get_local_id(0) % NUM_REDUCE_STAGES);
    const int warpId = static_cast<int>(item.get_local_id(0) / NUM_REDUCE_STAGES);
    const int warpsPerBlock = static_cast<int>(item.get_local_range(0) / NUM_REDUCE_STAGES);
    const int startWorkerId = static_cast<int>(item.get_group(0)) * warpsPerBlock + warpId;
    const int numWorkers = static_cast<int>(item.get_group_range(0)) * warpsPerBlock;

    const int totalQKHeads = num_heads_q + num_heads_k;
    const int totalWarps = num_tokens * totalQKHeads;

    // Grid-stride loop: when the launch is persistent (fewer work-groups than
    // totalWarps would need for a 1:1 mapping), each sub-group processes
    // multiple (token, head) pairs in sequence instead of exactly one.
    for (int globalWarpId = startWorkerId; globalWarpId < totalWarps; globalWarpId += numWorkers) {
      const int tokenIdx = globalWarpId / totalQKHeads;
      const int headIdx = globalWarpId % totalQKHeads;
      const bool isQ = headIdx < num_heads_q;
      const int localHeadIdx = isQ ? headIdx : (headIdx - num_heads_q);

      const int64_t totalHeads = static_cast<int64_t>(num_heads_q) + num_heads_k + num_heads_v;
      const int64_t rowStride = totalHeads * head_dim;
      const int64_t kColOffset = static_cast<int64_t>(num_heads_q) * head_dim;
      const int64_t headColOffset = isQ ? static_cast<int64_t>(localHeadIdx) * head_dim
                                        : kColOffset + static_cast<int64_t>(localHeadIdx) * head_dim;
      const int64_t rowBase = static_cast<int64_t>(tokenIdx) * rowStride + headColOffset;
      const int64_t offsetThread = rowBase + static_cast<int64_t>(laneId) * numElemsPerThread;

      accscalar_t elements[numElemsPerThread];
      accscalar_t sumOfSquares = 0;
#pragma unroll
      for (int c = 0; c < numChunks; c++) {
        const VecT in_vec = *reinterpret_cast<const VecT*>(qkv + offsetThread + c * VecSize);
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
          accscalar_t val = static_cast<accscalar_t>(in_vec[v]);
          elements[c * VecSize + v] = val;
          sumOfSquares += val * val;
        }
      }

      // Reduce sum across sub-group (warp)
      sumOfSquares = subGroupReduceSum(sumOfSquares, sg);

      // Compute RMS normalization factor
      float rms_rcp = sycl::rsqrt(sumOfSquares / static_cast<float>(head_dim) + eps);

      // Normalize elements
      const scalar_t* weight_ptr = isQ ? q_weight : k_weight;
#pragma unroll
      for (int c = 0; c < numChunks; c++) {
        const VecT w_vec = *reinterpret_cast<const VecT*>(weight_ptr + laneId * numElemsPerThread + c * VecSize);
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
          accscalar_t weight = static_cast<accscalar_t>(w_vec[v]);
          elements[c * VecSize + v] *= rms_rcp * weight;
        }
      }

      // Apply RoPE to normalized elements
      accscalar_t elements2[numElemsPerThread];
      accscalar_t cos_vals[numElemsPerThread];
      accscalar_t sin_vals[numElemsPerThread];
      float pos_id = static_cast<float>(position_ids[tokenIdx]);
      const int rotary_lanes = rotary_dim / numElemsPerThread;
      const bool applyRotary = (laneId < rotary_lanes);

      if (applyRotary) {
        if constexpr (interleave) {
          // Interleave mode
          for (int i = 0; i < numElemsPerThread; i++) {
            elements2[i] = (i % 2 == 0) ? -elements[i + 1] : elements[i - 1];

            int dim_idx = laneId * numElemsPerThread + i;
            int half_dim = dim_idx / 2;
            float freq = compute_freq_yarn(base, rotary_dim, half_dim, factor, low, high);
            float theta = pos_id * freq;
            sin_vals[i] = sycl::sin(theta);
            cos_vals[i] = sycl::cos(theta);
          }
        } else {
          // Neox style - use XOR shuffle like CUDA
          sycl::group_barrier(sg);
          const int half_rotary_lanes = rotary_lanes / 2;

          for (int i = 0; i < numElemsPerThread; i++) {
            // XOR shuffle to exchange between first and second half
            elements2[i] = sycl::permute_group_by_xor(sg, elements[i], half_rotary_lanes);
            if (laneId < half_rotary_lanes) {
              elements2[i] = -elements2[i];
            }

            int dim_idx = laneId * numElemsPerThread + i;
            dim_idx = (dim_idx * 2) % rotary_dim;
            int half_dim = dim_idx / 2;
            float freq = compute_freq_yarn(base, rotary_dim, half_dim, factor, low, high);
            float theta = pos_id * freq;
            sin_vals[i] = sycl::sin(theta);
            cos_vals[i] = sycl::cos(theta);
          }
          sycl::group_barrier(sg);
        }

        // Apply rotation with attention_factor
        for (int i = 0; i < numElemsPerThread; i++) {
          elements[i] = (elements[i] * cos_vals[i] + elements2[i] * sin_vals[i]) * attention_factor;
        }
      }

      // Store results
#pragma unroll
      for (int c = 0; c < numChunks; c++) {
        VecT out_vec;
#pragma unroll
        for (int v = 0; v < VecSize; v++) {
          out_vec[v] = static_cast<scalar_t>(elements[c * VecSize + v]);
        }
        *reinterpret_cast<VecT*>(qkv + offsetThread + c * VecSize) = out_vec;
      }
    }
  }
};

template <int head_dim, bool interleave, typename scalar_t, int VecSize>
void launchFusedQKNormRopeVecImpl(
    void* qkv,
    int num_tokens,
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
    float eps,
    const void* q_weight,
    const void* k_weight,
    float base,
    const int* position_ids,
    float factor,
    float low,
    float high,
    float attention_factor,
    int rotary_dim,
    sycl::queue& q,
    int64_t gridSize,
    int64_t blockSize) {
  FusedQKNormRopeKernel<head_dim, interleave, scalar_t, VecSize> kernel{
      static_cast<scalar_t*>(qkv),
      num_heads_q,
      num_heads_k,
      num_heads_v,
      eps,
      static_cast<const scalar_t*>(q_weight),
      static_cast<const scalar_t*>(k_weight),
      base,
      position_ids,
      num_tokens,
      factor,
      low,
      high,
      attention_factor,
      rotary_dim};

  sycl_kernel_submit(sycl::range<1>(gridSize * blockSize), sycl::range<1>(blockSize), q, kernel);
}

template <int head_dim, bool interleave, typename scalar_t>
void launchFusedQKNormRopeImpl(
    void* qkv,
    int num_tokens,
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
    float eps,
    const void* q_weight,
    const void* k_weight,
    float base,
    const int* position_ids,
    float factor,
    float low,
    float high,
    float attention_factor,
    int rotary_dim,
    sycl::queue& q) {
  const int totalQKHeads = num_heads_q + num_heads_k;
  const int64_t totalWarps = static_cast<int64_t>(num_tokens) * totalQKHeads;
  const auto launch_cfg = computePersistentLaunchConfig(totalWarps, NUM_REDUCE_STAGES);

  constexpr int64_t numElemsPerThread = head_dim / NUM_REDUCE_STAGES;
  const int64_t maxVecSize = std::min<int64_t>(numElemsPerThread, maxHwVecSize(sizeof(scalar_t)));
  const int64_t vec_size = pick_aligned_vec_size(maxVecSize, sizeof(scalar_t), {qkv, q_weight, k_weight});

  dispatchFusedQKNormRopeVecSize<numElemsPerThread>(vec_size, [&](auto vec_size_tag) {
    constexpr int kVecSize = static_cast<int>(decltype(vec_size_tag)::value);
    launchFusedQKNormRopeVecImpl<head_dim, interleave, scalar_t, kVecSize>(
        qkv,
        num_tokens,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        eps,
        q_weight,
        k_weight,
        base,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
        q,
        launch_cfg.gridSize,
        launch_cfg.blockSize);
  });
}

void fused_qk_norm_rope(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    double base,
    bool is_neox,
    torch::Tensor& position_ids,
    double factor,
    double low,
    double high,
    double attention_factor,
    int64_t rotary_dim) {
  // Input validation
  TORCH_CHECK(qkv.dim() == 2, "QKV tensor must be 2D: [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]");
  TORCH_CHECK(position_ids.dim() == 1, "Position IDs must be 1D: [num_tokens]");
  TORCH_CHECK(q_weight.dim() == 1, "Query weights must be 1D: [head_dim]");
  TORCH_CHECK(k_weight.dim() == 1, "Key weights must be 1D: [head_dim]");
  TORCH_CHECK(q_weight.size(0) == head_dim, "Query weights size must match head dimension");
  TORCH_CHECK(k_weight.size(0) == head_dim, "Key weights size must match head dimension");
  TORCH_CHECK(rotary_dim % (head_dim / NUM_REDUCE_STAGES) == 0, "rotary_dim must be divisible by numElemsPerThread");

  if (is_neox) {
    int64_t half_rotary_lanes = rotary_dim / (head_dim / NUM_REDUCE_STAGES) / 2;
    TORCH_CHECK(
        half_rotary_lanes >= 1 && (half_rotary_lanes & (half_rotary_lanes - 1)) == 0,
        "half_rotary_lanes must be a power of 2 for neox style, got ",
        half_rotary_lanes);
  }

  check_subgroup_size_supported(NUM_REDUCE_STAGES);

  CHECK_DEVICE(qkv);
  CHECK_CONTIGUOUS(qkv);
  CHECK_DEVICE(position_ids);
  CHECK_CONTIGUOUS(position_ids);
  TORCH_CHECK(
      position_ids.scalar_type() == at::ScalarType::Int,
      "position_ids must have dtype int32 (at::kInt); got ",
      position_ids.scalar_type());
  CHECK_DEVICE(q_weight);
  CHECK_CONTIGUOUS(q_weight);
  CHECK_DEVICE(k_weight);
  CHECK_CONTIGUOUS(k_weight);

  int64_t num_tokens = qkv.size(0);
  TORCH_CHECK(position_ids.size(0) == num_tokens, "Number of tokens in position_ids must match QKV");

  int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
  TORCH_CHECK(
      qkv.size(1) == total_heads * head_dim, "QKV tensor size must match total number of heads and head dimension");

  auto queue = dpcppGetCurrentQueue();
  bool interleave = !is_neox;

  dispatchFusedQKNormRopeScalarType<true>(qkv.scalar_type(), "fused_qk_norm_rope", [&](auto scalar_tag) {
    using scalar_t = typename decltype(scalar_tag)::type;
    dispatchFusedQKNormRopeHeadDim(head_dim, "fusedQKNormRope", [&](auto head_dim_tag) {
      constexpr int64_t kHeadDimConst = decltype(head_dim_tag)::value;
      if (interleave) {
        launchFusedQKNormRopeImpl<kHeadDimConst, true, scalar_t>(
            qkv.data_ptr(),
            static_cast<int>(num_tokens),
            static_cast<int>(num_heads_q),
            static_cast<int>(num_heads_k),
            static_cast<int>(num_heads_v),
            static_cast<float>(eps),
            q_weight.data_ptr(),
            k_weight.data_ptr(),
            static_cast<float>(base),
            position_ids.data_ptr<int>(),
            static_cast<float>(factor),
            static_cast<float>(low),
            static_cast<float>(high),
            static_cast<float>(attention_factor),
            static_cast<int>(rotary_dim),
            queue);
      } else {
        launchFusedQKNormRopeImpl<kHeadDimConst, false, scalar_t>(
            qkv.data_ptr(),
            static_cast<int>(num_tokens),
            static_cast<int>(num_heads_q),
            static_cast<int>(num_heads_k),
            static_cast<int>(num_heads_v),
            static_cast<float>(eps),
            q_weight.data_ptr(),
            k_weight.data_ptr(),
            static_cast<float>(base),
            position_ids.data_ptr<int>(),
            static_cast<float>(factor),
            static_cast<float>(low),
            static_cast<float>(high),
            static_cast<float>(attention_factor),
            static_cast<int>(rotary_dim),
            queue);
      }
    });
  });
}

// SYCL Kernel for Fused QK Norm + RoPE using a precomputed cos/sin cache
// (mirrors CUDA's qknorm_rope.cuh). q/k must be 3D contiguous tensors:
// [num_tokens, num_heads, head_dim].
template <int64_t kHeadDim, bool kIsNeox, typename scalar_t, typename IdType, int64_t kVecSize>
struct FusedQKNormRopeCacheKernel {
  static_assert(kHeadDim <= 256, "Only head_dim <= 256 is supported");
  static_assert(
      kHeadDim % NUM_REDUCE_STAGES == 0, "head_dim must be divisible by the sub-group size (NUM_REDUCE_STAGES)");

  static constexpr uint32_t kElemsPerThread = static_cast<uint32_t>(kHeadDim / NUM_REDUCE_STAGES);
  static_assert(kElemsPerThread % kVecSize == 0, "kVecSize must evenly divide kElemsPerThread");
  static constexpr uint32_t kNumChunks = kElemsPerThread / kVecSize;

  scalar_t* q_ptr;
  scalar_t* k_ptr;
  const scalar_t* q_weight_ptr;
  const scalar_t* k_weight_ptr;
  const float* cos_sin_cache_ptr;  // [max_position, kRopeDim]
  const IdType* positions;         // [num_tokens]
  int64_t rope_dim;
  int64_t rotary_lanes;
  int64_t half_rotary_lanes;
  int64_t q_token_stride;  // elements between consecutive tokens in q (== num_qo_heads * kHeadDim)
  int64_t k_token_stride;  // elements between consecutive tokens in k (== num_kv_heads * kHeadDim)
  uint32_t num_qo_heads;
  uint32_t num_kv_heads;
  uint32_t num_tokens;
  float eps;

  [[sycl::reqd_sub_group_size(NUM_REDUCE_STAGES)]] void operator()(sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    const uint32_t lane_id = static_cast<uint32_t>(item.get_local_id(0) % NUM_REDUCE_STAGES);
    const uint32_t warp_id = static_cast<uint32_t>(item.get_local_id(0) / NUM_REDUCE_STAGES);
    const uint32_t warps_per_block = static_cast<uint32_t>(item.get_local_range(0) / NUM_REDUCE_STAGES);
    const uint32_t start_worker_id = static_cast<uint32_t>(item.get_group(0)) * warps_per_block + warp_id;
    const uint32_t num_workers = static_cast<uint32_t>(item.get_group_range(0)) * warps_per_block;

    const uint32_t num_qk_heads = num_qo_heads + num_kv_heads;
    const uint32_t num_works = num_qk_heads * num_tokens;

    for (uint32_t idx = start_worker_id; idx < num_works; idx += num_workers) {
      const uint32_t token_id = idx / num_qk_heads;
      const uint32_t head_id = idx % num_qk_heads;
      const bool load_q = head_id < num_qo_heads;

      scalar_t* row_ptr;
      const scalar_t* weight_ptr;
      if (load_q) {
        row_ptr = q_ptr + static_cast<int64_t>(token_id) * q_token_stride + static_cast<int64_t>(head_id) * kHeadDim;
        weight_ptr = q_weight_ptr;
      } else {
        const uint32_t k_head_id = head_id - num_qo_heads;
        row_ptr = k_ptr + static_cast<int64_t>(token_id) * k_token_stride + static_cast<int64_t>(k_head_id) * kHeadDim;
        weight_ptr = k_weight_ptr;
      }

      using VecT = aligned_vector_loop<scalar_t, kVecSize>;
      float elems[kElemsPerThread];
      float sum_of_squares = 0.0f;
#pragma unroll
      for (uint32_t c = 0; c < kNumChunks; ++c) {
        const VecT in_vec = *reinterpret_cast<const VecT*>(row_ptr + lane_id * kElemsPerThread + c * kVecSize);
#pragma unroll
        for (uint32_t v = 0; v < kVecSize; ++v) {
          const float x = static_cast<float>(in_vec[v]);
          elems[c * kVecSize + v] = x;
          sum_of_squares += x * x;
        }
      }

      sum_of_squares = subGroupReduceSum(sum_of_squares, sg);
      const float norm_factor = sycl::rsqrt(sum_of_squares / static_cast<float>(kHeadDim) + eps);

#pragma unroll
      for (uint32_t c = 0; c < kNumChunks; ++c) {
        const VecT w_vec = *reinterpret_cast<const VecT*>(weight_ptr + lane_id * kElemsPerThread + c * kVecSize);
#pragma unroll
        for (uint32_t v = 0; v < kVecSize; ++v) {
          const float w = static_cast<float>(w_vec[v]);
          elems[c * kVecSize + v] *= norm_factor * w;
        }
      }

      const int64_t pos = static_cast<int64_t>(positions[token_id]);
      const float* cos_ptr = cos_sin_cache_ptr + pos * rope_dim;
      const float* sin_ptr = cos_ptr + rope_dim / 2;
      const bool apply_rotary = static_cast<int64_t>(lane_id) < rotary_lanes;

      if constexpr (kIsNeox) {
        sycl::group_barrier(sg);
        float permuted[kElemsPerThread];
#pragma unroll
        for (uint32_t i = 0; i < kElemsPerThread; ++i) {
          permuted[i] = sycl::permute_group_by_xor(sg, elems[i], static_cast<int>(half_rotary_lanes));
        }
        sycl::group_barrier(sg);
        if (apply_rotary) {
#pragma unroll
          for (uint32_t i = 0; i < kElemsPerThread; ++i) {
            float swapped = permuted[i];
            if (static_cast<int64_t>(lane_id) < half_rotary_lanes) {
              swapped = -swapped;
            }

            int dim_idx = static_cast<int>(lane_id * kElemsPerThread + i);
            dim_idx = (dim_idx * 2) % static_cast<int>(rope_dim);
            const int half_idx = dim_idx / 2;
            const float cos = cos_ptr[half_idx];
            const float sin = sin_ptr[half_idx];
            elems[i] = elems[i] * cos + swapped * sin;
          }
        }
      } else {
        if (apply_rotary) {
#pragma unroll
          for (uint32_t i = 0; i < kElemsPerThread; i += 2) {
            const float x = elems[i];
            const float y = elems[i + 1];
            const int half_idx = static_cast<int>(lane_id * kElemsPerThread + i) / 2;
            const float cos = cos_ptr[half_idx];
            const float sin = sin_ptr[half_idx];
            elems[i] = x * cos - y * sin;
            elems[i + 1] = y * cos + x * sin;
          }
        }
      }

#pragma unroll
      for (uint32_t c = 0; c < kNumChunks; ++c) {
        VecT out_vec;
#pragma unroll
        for (uint32_t v = 0; v < kVecSize; ++v) {
          out_vec[v] = static_cast<scalar_t>(elems[c * kVecSize + v]);
        }
        *reinterpret_cast<VecT*>(row_ptr + lane_id * kElemsPerThread + c * kVecSize) = out_vec;
      }
    }
  }
};

template <int64_t kHeadDim, bool kIsNeox, typename scalar_t, typename IdType, int64_t kVecSize>
void launchFusedQKNormRopeCacheVecImpl(
    scalar_t* q_ptr,
    scalar_t* k_ptr,
    const scalar_t* q_weight_ptr,
    const scalar_t* k_weight_ptr,
    const float* cos_sin_cache_ptr,
    const IdType* positions_ptr,
    int64_t q_token_stride,
    int64_t k_token_stride,
    int64_t num_tokens,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    int64_t rope_dim,
    int64_t rotary_lanes,
    int64_t half_rotary_lanes,
    float eps,
    sycl::queue& queue,
    int64_t gridSize,
    int64_t blockSize) {
  using KernelT = FusedQKNormRopeCacheKernel<kHeadDim, kIsNeox, scalar_t, IdType, kVecSize>;
  KernelT kernel{
      q_ptr,
      k_ptr,
      q_weight_ptr,
      k_weight_ptr,
      cos_sin_cache_ptr,
      positions_ptr,
      rope_dim,
      rotary_lanes,
      half_rotary_lanes,
      q_token_stride,
      k_token_stride,
      static_cast<uint32_t>(num_qo_heads),
      static_cast<uint32_t>(num_kv_heads),
      static_cast<uint32_t>(num_tokens),
      eps};

  sycl_kernel_submit(sycl::range<1>(gridSize * blockSize), sycl::range<1>(blockSize), queue, kernel);
}

template <int64_t kHeadDim, bool kIsNeox, typename scalar_t, typename IdType>
void launchFusedQKNormRopeCacheImpl(
    scalar_t* q_ptr,
    scalar_t* k_ptr,
    const scalar_t* q_weight_ptr,
    const scalar_t* k_weight_ptr,
    const float* cos_sin_cache_ptr,
    const IdType* positions_ptr,
    int64_t q_token_stride,
    int64_t k_token_stride,
    int64_t num_tokens,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    int64_t rope_dim,
    float eps,
    sycl::queue& queue) {
  const int64_t totalWork = num_tokens * (num_qo_heads + num_kv_heads);
  const auto launch_cfg = computePersistentLaunchConfig(totalWork, NUM_REDUCE_STAGES);

  constexpr int64_t kElemsPerThread = kHeadDim / NUM_REDUCE_STAGES;
  TORCH_CHECK(rope_dim > 0 && rope_dim <= kHeadDim, "Invalid rope_dim: ", rope_dim);
  TORCH_CHECK(rope_dim % kElemsPerThread == 0, "rope_dim must align with per-lane vector width");
  const int64_t rotary_lanes = rope_dim / kElemsPerThread;
  const int64_t half_rotary_lanes = rotary_lanes / 2;
  if constexpr (kIsNeox) {
    TORCH_CHECK(
        rotary_lanes >= 2 && (rotary_lanes & (rotary_lanes - 1)) == 0,
        "NeoX fused qknorm+rope requires rotary lane count to be a power of 2, got ",
        rotary_lanes);
  }
  const int64_t maxVecSize = std::min<int64_t>(kElemsPerThread, maxHwVecSize(sizeof(scalar_t)));
  const int64_t vec_size =
      pick_aligned_vec_size(maxVecSize, sizeof(scalar_t), {q_ptr, k_ptr, q_weight_ptr, k_weight_ptr});

  dispatchFusedQKNormRopeVecSize<kElemsPerThread>(vec_size, [&](auto vec_size_tag) {
    constexpr int64_t kVecSize = decltype(vec_size_tag)::value;
    launchFusedQKNormRopeCacheVecImpl<kHeadDim, kIsNeox, scalar_t, IdType, kVecSize>(
        q_ptr,
        k_ptr,
        q_weight_ptr,
        k_weight_ptr,
        cos_sin_cache_ptr,
        positions_ptr,
        q_token_stride,
        k_token_stride,
        num_tokens,
        num_qo_heads,
        num_kv_heads,
        rope_dim,
        rotary_lanes,
        half_rotary_lanes,
        eps,
        queue,
        launch_cfg.gridSize,
        launch_cfg.blockSize);
  });
}

void fused_qk_norm_rope_with_cos_sin_cache_inplace(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    torch::Tensor& cos_sin_cache,
    torch::Tensor& positions,
    bool is_neox,
    double eps) {
  TORCH_CHECK(q.dim() == k.dim(), "q and k must have the same rank, got q:", q.dim(), " k:", k.dim());
  TORCH_CHECK(q.dim() == 3 || q.dim() == 4, "q/k must be 3D or 4D tensors, got q:", q.dim());
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must have the same dtype");
  TORCH_CHECK(q_weight.scalar_type() == q.scalar_type(), "q_weight dtype must match q dtype");
  TORCH_CHECK(k_weight.scalar_type() == k.scalar_type(), "k_weight dtype must match k dtype");
  TORCH_CHECK(cos_sin_cache.scalar_type() == at::ScalarType::Float, "cos_sin_cache must be float32");

  CHECK_DEVICE(q);
  CHECK_CONTIGUOUS(q);
  CHECK_DEVICE(k);
  CHECK_CONTIGUOUS(k);
  CHECK_DEVICE(q_weight);
  CHECK_CONTIGUOUS(q_weight);
  CHECK_DEVICE(k_weight);
  CHECK_CONTIGUOUS(k_weight);
  CHECK_DEVICE(cos_sin_cache);
  CHECK_CONTIGUOUS(cos_sin_cache);
  CHECK_DEVICE(positions);
  CHECK_CONTIGUOUS(positions);

  auto q_view = q.dim() == 4 ? q.view({-1, q.size(2), q.size(3)}) : q;
  auto k_view = k.dim() == 4 ? k.view({-1, k.size(2), k.size(3)}) : k;
  TORCH_CHECK(q_view.dim() == 3 && k_view.dim() == 3, "Flattened q/k must be 3D tensors");
  TORCH_CHECK(q_view.size(0) == k_view.size(0), "q and k must have the same token count after flattening");
  TORCH_CHECK(q_view.size(2) == k_view.size(2), "q and k must have the same head_dim");

  const int64_t num_tokens = q_view.size(0);
  const int64_t num_qo_heads = q_view.size(1);
  const int64_t num_kv_heads = k_view.size(1);
  const int64_t head_dim = q_view.size(2);

  TORCH_CHECK(q_weight.dim() == 1, "q_weight must be 1D [head_dim]");
  TORCH_CHECK(k_weight.dim() == 1, "k_weight must be 1D [head_dim]");
  TORCH_CHECK(q_weight.size(0) == head_dim, "q_weight size must match head_dim");
  TORCH_CHECK(k_weight.size(0) == head_dim, "k_weight size must match head_dim");
  TORCH_CHECK(cos_sin_cache.dim() == 2, "cos_sin_cache must be 2D [max_position, rope_dim]");
  const int64_t rope_dim = cos_sin_cache.size(1);
  TORCH_CHECK(rope_dim % 2 == 0, "rope_dim must be even");
  TORCH_CHECK(rope_dim <= head_dim, "rope_dim must be <= head_dim");
  TORCH_CHECK(positions.dim() == 1, "positions must be 1D [num_tokens]");
  TORCH_CHECK(positions.size(0) == num_tokens, "positions size must match flattened q/k tokens");

  check_subgroup_size_supported(NUM_REDUCE_STAGES);

  auto queue = dpcppGetCurrentQueue();

  dispatchFusedQKNormRopeScalarType<false>(
      q_view.scalar_type(), "fused_qk_norm_rope_with_cos_sin_cache_inplace", [&](auto scalar_tag) {
        using scalar_t = typename decltype(scalar_tag)::type;
        dispatchFusedQKNormRopePositionsType(
            positions.scalar_type(), "fused_qk_norm_rope_with_cos_sin_cache_inplace", [&](auto id_tag) {
              using IdType = typename decltype(id_tag)::type;
              dispatchFusedQKNormRopeHeadDim(
                  head_dim, "fused_qk_norm_rope_with_cos_sin_cache_inplace", [&](auto head_dim_tag) {
                    constexpr int64_t kHeadDimConst = decltype(head_dim_tag)::value;
                    if (is_neox) {
                      launchFusedQKNormRopeCacheImpl<kHeadDimConst, true, scalar_t, IdType>(
                          static_cast<scalar_t*>(q_view.data_ptr()),
                          static_cast<scalar_t*>(k_view.data_ptr()),
                          static_cast<const scalar_t*>(q_weight.data_ptr()),
                          static_cast<const scalar_t*>(k_weight.data_ptr()),
                          static_cast<const float*>(cos_sin_cache.data_ptr()),
                          static_cast<const IdType*>(positions.data_ptr()),
                          q_view.stride(0),
                          k_view.stride(0),
                          num_tokens,
                          num_qo_heads,
                          num_kv_heads,
                          rope_dim,
                          static_cast<float>(eps),
                          queue);
                    } else {
                      launchFusedQKNormRopeCacheImpl<kHeadDimConst, false, scalar_t, IdType>(
                          static_cast<scalar_t*>(q_view.data_ptr()),
                          static_cast<scalar_t*>(k_view.data_ptr()),
                          static_cast<const scalar_t*>(q_weight.data_ptr()),
                          static_cast<const scalar_t*>(k_weight.data_ptr()),
                          static_cast<const float*>(cos_sin_cache.data_ptr()),
                          static_cast<const IdType*>(positions.data_ptr()),
                          q_view.stride(0),
                          k_view.stride(0),
                          num_tokens,
                          num_qo_heads,
                          num_kv_heads,
                          rope_dim,
                          static_cast<float>(eps),
                          queue);
                    }
                  });
            });
      });
}

}  // namespace at::native::xpu
