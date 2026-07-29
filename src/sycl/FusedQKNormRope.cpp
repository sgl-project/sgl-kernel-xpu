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

// Sub-group reduction for sum
template <typename T>
inline T subGroupReduceSum(T val, const sycl::sub_group& sg) {
  return sycl::reduce_over_group(sg, val, sycl::plus<T>());
}

// Uses exp2(x * log2_base) instead of pow(base, x): exp2 lowers to a single
// hardware instruction on Intel GPUs, whereas pow goes through a slow
// polynomial path. Callers precompute log2_base = log2(base) once on the
// host (see launchFusedQKNormRopeImpl below) instead of passing base.
inline float computeFreqYarn(float log2_base, int rotary_dim, int half_dim, float factor, float low, float high) {
  const float exponent = -2.0f * static_cast<float>(half_dim) / static_cast<float>(rotary_dim);
  float freq = sycl::exp2(exponent * log2_base);

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
    case at::ScalarType::Float:
      fn(TypeTag<float>{});
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

// Dispatches over vectorized VecSize instantiations (16/8/4/2), accepting
// only widths that both fit within the per-lane element count and divide it
// evenly, then unconditionally falling back to scalar (VecSize=1) -- always
// valid regardless of alignment -- if none matched, guaranteeing the kernel
// is always launched.
template <int64_t kElemsPerThread, typename Fn>
inline void dispatchFusedQKNormRopeVecSize(int64_t vec_size, Fn&& fn) {
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

// Picks the largest power-of-two width (<= max_vec_size) for which:
//   1. `elems_per_thread` (the per-lane algorithmic element count) is evenly
//      divisible by vec_size -- required so the kernel's static_assert
//      (kElemsPerThread % VecSize == 0) always holds;
//   2. every pointer in `ptrs` is aligned to `elem_size * vec_size` bytes;
//   3. every stride in `elem_strides` (elements; row strides for
//      non-contiguous q/k) is itself a multiple of vec_size.
// Falls back to 1 (always valid) if no larger width satisfies all three.
// `elem_strides` defaults to empty for fully contiguous callers. Only the
// physical load/store chunk width shrinks; `elems_per_thread` itself is
// unaffected.
inline int64_t pickAlignedVecSize(
    int64_t max_vec_size,
    int64_t elem_size,
    std::initializer_list<const void*> ptrs,
    int64_t elems_per_thread,
    std::initializer_list<int64_t> elem_strides = {}) {
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
    if (ok) {
      for (int64_t stride : elem_strides) {
        if (stride % vec_size != 0) {
          ok = false;
          break;
        }
      }
    }
    if (ok) return vec_size;
  }
  return 1;
}

// Picks the device's preferred vector width, clamped by `pickAlignedVecSize`
// for the row's element count/alignment/strides. Shared preamble for every
// norm+rope launcher below.
inline int64_t pickVecSizeForRow(
    int64_t elem_size,
    std::initializer_list<const void*> ptrs,
    int64_t elems_per_thread,
    std::initializer_list<int64_t> elem_strides = {}) {
  const int64_t preferredVecSize = preferred_vector_width(dpcppGetDeviceIdOfCurrentQueue(), elem_size);
  const int64_t maxVecSize = std::min<int64_t>(elems_per_thread, preferredVecSize);
  return pickAlignedVecSize(maxVecSize, elem_size, ptrs, elems_per_thread, elem_strides);
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

// Shared row addressing: base + token_id*token_stride + head_id*head_stride.
template <typename T>
inline T* rowPtr(T* base, int64_t token_id, int64_t head_id, int64_t token_stride, int64_t head_stride) {
  return base + token_id * token_stride + head_id * head_stride;
}

// (x_re + i*x_im) * (f_re + i*f_im); outputs may alias the inputs.
inline void ropeRotate(float x_re, float x_im, float f_re, float f_im, float& out_re, float& out_im) {
  out_re = x_re * f_re - x_im * f_im;
  out_im = x_re * f_im + x_im * f_re;
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
  float log2_base;
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
            float freq = computeFreqYarn(log2_base, rotary_dim, half_dim, factor, low, high);
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
            float freq = computeFreqYarn(log2_base, rotary_dim, half_dim, factor, low, high);
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
    float log2_base,
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
      log2_base,
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

  // Precompute log2(base) once per launch instead of per-thread (see
  // compute_freq_yarn above).
  const float log2_base = std::log2(base);

  constexpr int64_t numElemsPerThread = head_dim / NUM_REDUCE_STAGES;
  const int64_t vec_size = pickVecSizeForRow(sizeof(scalar_t), {qkv, q_weight, k_weight}, numElemsPerThread);

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
        log2_base,
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
// (mirrors CUDA's qknorm_rope.cuh). q/k must be 3D tensors (after flattening
// any leading batch/seq dims): [num_tokens, num_heads, head_dim]. Only the
// last dimension (head_dim) is required to be contiguous; the token and head
// strides may be arbitrary (e.g. q/k sliced out of a larger packed buffer),
// so they are passed in explicitly rather than assumed to equal
// num_heads * head_dim / head_dim.
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
  int64_t q_token_stride;  // elements between consecutive tokens in q
  int64_t k_token_stride;  // elements between consecutive tokens in k
  int64_t q_head_stride;   // elements between consecutive heads in q (may differ from kHeadDim)
  int64_t k_head_stride;   // elements between consecutive heads in k (may differ from kHeadDim)
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
        row_ptr =
            rowPtr(q_ptr, static_cast<int64_t>(token_id), static_cast<int64_t>(head_id), q_token_stride, q_head_stride);
        weight_ptr = q_weight_ptr;
      } else {
        const uint32_t k_head_id = head_id - num_qo_heads;
        row_ptr = rowPtr(
            k_ptr, static_cast<int64_t>(token_id), static_cast<int64_t>(k_head_id), k_token_stride, k_head_stride);
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
            const int half_idx = static_cast<int>(lane_id * kElemsPerThread + i) / 2;
            ropeRotate(elems[i], elems[i + 1], cos_ptr[half_idx], sin_ptr[half_idx], elems[i], elems[i + 1]);
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
    int64_t q_head_stride,
    int64_t k_head_stride,
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
      q_head_stride,
      k_head_stride,
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
    int64_t q_head_stride,
    int64_t k_head_stride,
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
  // q/k rows may be non-contiguous beyond the last dimension (e.g. sliced
  // out of a larger packed buffer), so the chosen vector width must also
  // divide the token/head strides, not just satisfy base-pointer alignment.
  const int64_t vec_size = pickVecSizeForRow(
      sizeof(scalar_t),
      {q_ptr, k_ptr, q_weight_ptr, k_weight_ptr},
      kElemsPerThread,
      {q_token_stride, k_token_stride, q_head_stride, k_head_stride});

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
        q_head_stride,
        k_head_stride,
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

// ============================================================================
// Fused Q-only RMSNorm-self (no weight) + RoPE for DeepSeek-V4's Q path
// (`fused_q_norm_rope`), reading an *interleaved* freqs_cis table
// [max_position, rope_dim] ([re0, im0, re1, im1, ...]). Differs from
// `fused_inplace_qknorm_rope` above: no learned weight, and RoPE applies to
// the *last* rope_dim elements (DSV4's [nope | rope] layout).
//
// Two paths, dispatched on head_dim:
//   - Warp: one sub-group per (token, head) row. Only for head_dim in
//     {64, 128, 192, 256} with rope_dim aligned to a lane boundary (so the
//     real/imag pair always lands within one lane -- no cross-lane traffic).
//   - CTA: one work-group per row, staged through local memory so the
//     real/imag pairing works for any head_dim/rope_dim (e.g. DSV4's 512).
// ============================================================================

template <int64_t kHeadDim, typename scalar_t, typename IdType, int64_t kVecSize>
struct FusedQNormRopeWarpKernel {
  static_assert(kHeadDim % NUM_REDUCE_STAGES == 0, "head_dim must be divisible by NUM_REDUCE_STAGES");
  static constexpr uint32_t kElemsPerThread = static_cast<uint32_t>(kHeadDim / NUM_REDUCE_STAGES);
  static_assert(kElemsPerThread % kVecSize == 0, "kVecSize must evenly divide kElemsPerThread");
  static_assert(kElemsPerThread % 2 == 0, "kElemsPerThread must be even for real/imag pairing");
  static constexpr uint32_t kNumChunks = kElemsPerThread / kVecSize;

  const scalar_t* q_ptr;
  scalar_t* out_ptr;
  const float* freqs_cis_ptr;  // [max_position, rope_dim], interleaved re/im
  const IdType* positions;     // [num_tokens]
  int64_t q_token_stride;
  int64_t q_head_stride;
  int64_t out_token_stride;
  int64_t out_head_stride;
  int64_t rope_dim;
  uint32_t num_heads;
  uint32_t num_tokens;
  float eps;

  [[sycl::reqd_sub_group_size(NUM_REDUCE_STAGES)]] void operator()(sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    const uint32_t lane_id = static_cast<uint32_t>(item.get_local_id(0) % NUM_REDUCE_STAGES);
    const uint32_t warp_id = static_cast<uint32_t>(item.get_local_id(0) / NUM_REDUCE_STAGES);
    const uint32_t warps_per_block = static_cast<uint32_t>(item.get_local_range(0) / NUM_REDUCE_STAGES);
    const uint32_t start_worker_id = static_cast<uint32_t>(item.get_group(0)) * warps_per_block + warp_id;
    const uint32_t num_workers = static_cast<uint32_t>(item.get_group_range(0)) * warps_per_block;

    const uint32_t num_works = num_tokens * num_heads;

    // rope_dim % kElemsPerThread == 0 is enforced by the host launcher before
    // selecting this path, so nope_dim is also a multiple of kElemsPerThread
    // and every lane's contiguous block is either fully nope or fully rope.
    const int64_t nope_dim = kHeadDim - rope_dim;
    const int64_t rotary_lanes = rope_dim / static_cast<int64_t>(kElemsPerThread);
    const bool is_rope_lane = static_cast<int64_t>(lane_id) >= (NUM_REDUCE_STAGES - rotary_lanes);

    for (uint32_t idx = start_worker_id; idx < num_works; idx += num_workers) {
      const uint32_t token_id = idx / num_heads;
      const uint32_t head_id = idx % num_heads;

      const scalar_t* row_in =
          rowPtr(q_ptr, static_cast<int64_t>(token_id), static_cast<int64_t>(head_id), q_token_stride, q_head_stride);
      scalar_t* row_out = rowPtr(
          out_ptr, static_cast<int64_t>(token_id), static_cast<int64_t>(head_id), out_token_stride, out_head_stride);

      using VecT = aligned_vector_loop<scalar_t, kVecSize>;
      float elems[kElemsPerThread];
      float sum_of_squares = 0.0f;
#pragma unroll
      for (uint32_t c = 0; c < kNumChunks; ++c) {
        const VecT in_vec = *reinterpret_cast<const VecT*>(row_in + lane_id * kElemsPerThread + c * kVecSize);
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
      for (uint32_t i = 0; i < kElemsPerThread; ++i) {
        elems[i] *= norm_factor;
      }

      if (is_rope_lane) {
        const int64_t pos = static_cast<int64_t>(positions[token_id]);
        const float* freq_row = freqs_cis_ptr + pos * rope_dim;
        // First owned element's rope-local offset (>=0 and even, by the
        // alignment precondition above).
        const int64_t lane_rope_base = static_cast<int64_t>(lane_id) * kElemsPerThread - nope_dim;
#pragma unroll
        for (uint32_t i = 0; i < kElemsPerThread; i += 2) {
          const int64_t p = lane_rope_base + i;
          ropeRotate(elems[i], elems[i + 1], freq_row[p], freq_row[p + 1], elems[i], elems[i + 1]);
        }
      }

#pragma unroll
      for (uint32_t c = 0; c < kNumChunks; ++c) {
        VecT out_vec;
#pragma unroll
        for (uint32_t v = 0; v < kVecSize; ++v) {
          out_vec[v] = static_cast<scalar_t>(elems[c * kVecSize + v]);
        }
        *reinterpret_cast<VecT*>(row_out + lane_id * kElemsPerThread + c * kVecSize) = out_vec;
      }
    }
  }
};

template <int64_t kHeadDim, typename scalar_t, typename IdType>
void launchFusedQNormRopeWarpImpl(
    const scalar_t* q_ptr,
    scalar_t* out_ptr,
    const float* freqs_cis_ptr,
    const IdType* positions,
    int64_t q_token_stride,
    int64_t q_head_stride,
    int64_t out_token_stride,
    int64_t out_head_stride,
    int64_t rope_dim,
    int64_t num_tokens,
    int64_t num_heads,
    float eps,
    sycl::queue& queue) {
  const int64_t totalWork = num_tokens * num_heads;
  const auto launch_cfg = computePersistentLaunchConfig(totalWork, NUM_REDUCE_STAGES);

  constexpr int64_t kElemsPerThread = kHeadDim / NUM_REDUCE_STAGES;
  const int64_t vec_size = pickVecSizeForRow(
      sizeof(scalar_t),
      {q_ptr, out_ptr},
      kElemsPerThread,
      {q_token_stride, q_head_stride, out_token_stride, out_head_stride});

  dispatchFusedQKNormRopeVecSize<kElemsPerThread>(vec_size, [&](auto vec_size_tag) {
    constexpr int64_t kVecSize = decltype(vec_size_tag)::value;
    using KernelT = FusedQNormRopeWarpKernel<kHeadDim, scalar_t, IdType, kVecSize>;
    KernelT kernel{
        q_ptr,
        out_ptr,
        freqs_cis_ptr,
        positions,
        q_token_stride,
        q_head_stride,
        out_token_stride,
        out_head_stride,
        rope_dim,
        static_cast<uint32_t>(num_heads),
        static_cast<uint32_t>(num_tokens),
        eps};
    sycl_kernel_submit(
        sycl::range<1>(launch_cfg.gridSize * launch_cfg.blockSize),
        sycl::range<1>(launch_cfg.blockSize),
        queue,
        kernel);
  });
}

// CTA path: one work-group per (token, head) row (grid-stride when rows
// exceed resident work-groups). head_dim/rope_dim are runtime values.
template <typename scalar_t, typename IdType, int64_t kVecSize>
struct FusedQNormRopeCTAKernel {
  using VecT = aligned_vector_loop<scalar_t, kVecSize>;

  const scalar_t* q_ptr;
  scalar_t* out_ptr;
  const float* freqs_cis_ptr;
  const IdType* positions;
  int64_t q_token_stride;
  int64_t q_head_stride;
  int64_t out_token_stride;
  int64_t out_head_stride;
  int64_t head_dim;
  int64_t rope_dim;
  uint32_t num_heads;
  uint32_t num_tokens;
  float eps;
  sycl::local_accessor<float, 1> stage;  // [head_dim], raw (pre-norm) row values

  [[sycl::reqd_sub_group_size(NUM_REDUCE_STAGES)]] void operator()(sycl::nd_item<1> item) const {
    const int64_t local_id = static_cast<int64_t>(item.get_local_id(0));
    const int64_t workgroup_size = static_cast<int64_t>(item.get_local_range(0));
    const int64_t workgroup_id = static_cast<int64_t>(item.get_group(0));
    const int64_t num_workgroups = static_cast<int64_t>(item.get_group_range(0));
    const int64_t num_works = static_cast<int64_t>(num_tokens) * static_cast<int64_t>(num_heads);
    const int64_t nope_dim = head_dim - rope_dim;

    for (int64_t work_id = workgroup_id; work_id < num_works; work_id += num_workgroups) {
      const int64_t token_id = work_id / num_heads;
      const int64_t head_id = work_id % num_heads;

      const scalar_t* row_in = rowPtr(q_ptr, token_id, head_id, q_token_stride, q_head_stride);
      scalar_t* row_out = rowPtr(out_ptr, token_id, head_id, out_token_stride, out_head_stride);

      // Phase 1: load, stage into local memory, accumulate sum-of-squares.
      float sum_of_squares = 0.0f;
      for (int64_t dim = local_id * kVecSize; dim < head_dim; dim += workgroup_size * kVecSize) {
        const VecT in_vec = *reinterpret_cast<const VecT*>(row_in + dim);
#pragma unroll
        for (int64_t v = 0; v < kVecSize; ++v) {
          const float x = static_cast<float>(in_vec[v]);
          stage[dim + v] = x;
          sum_of_squares += x * x;
        }
      }

      sum_of_squares = sycl::reduce_over_group(item.get_group(), sum_of_squares, sycl::plus<float>());
      const float norm_factor = sycl::rsqrt(sum_of_squares / static_cast<float>(head_dim) + eps);

      // Wait for every lane to finish staging before reading neighbors for
      // RoPE pairing below.
      item.barrier(sycl::access::fence_space::local_space);

      const int64_t pos = static_cast<int64_t>(positions[token_id]);
      const float* freq_row = freqs_cis_ptr + pos * rope_dim;

      // Phase 2: read back, apply norm + RoPE, vectorized store.
      for (int64_t dim = local_id * kVecSize; dim < head_dim; dim += workgroup_size * kVecSize) {
        VecT out_vec;
#pragma unroll
        for (int64_t v = 0; v < kVecSize; ++v) {
          const int64_t abs_dim = dim + v;
          float x = stage[abs_dim] * norm_factor;
          if (abs_dim >= nope_dim) {
            const int64_t p = abs_dim - nope_dim;
            const bool is_real = (p % 2) == 0;
            const int64_t partner_abs_dim = is_real ? abs_dim + 1 : abs_dim - 1;
            const float partner = stage[partner_abs_dim] * norm_factor;
            const float x_re = is_real ? x : partner;
            const float x_im = is_real ? partner : x;
            const float f_re = freq_row[is_real ? p : p - 1];
            const float f_im = freq_row[is_real ? p + 1 : p];
            float out_re, out_im;
            ropeRotate(x_re, x_im, f_re, f_im, out_re, out_im);
            x = is_real ? out_re : out_im;
          }
          out_vec[v] = static_cast<scalar_t>(x);
        }
        *reinterpret_cast<VecT*>(row_out + dim) = out_vec;
      }

      // Wait before the next grid-stride iteration reuses `stage`.
      item.barrier(sycl::access::fence_space::local_space);
    }
  }
};

// Work-group size for the CTA path: sub-group-sized (16-lane) chunks that
// cover head_dim/vec_size elements, capped by the device max. Mirrors
// QKNormCommon.h's cta_workgroup_size.
inline int64_t computeQNormRopeCTAWorkgroupSize(int64_t head_dim, int64_t vec_size) {
  const int64_t max_wg_size =
      capWorkgroupSize(dpcppMaxWorkGroupSize(dpcppGetDeviceIdOfCurrentQueue()), NUM_REDUCE_STAGES);
  const int64_t plane_vecs = divUp(head_dim, vec_size);
  int64_t workgroup_size = divUp(plane_vecs, static_cast<int64_t>(NUM_REDUCE_STAGES)) * NUM_REDUCE_STAGES;
  workgroup_size = std::min(workgroup_size, max_wg_size);
  return std::max<int64_t>(workgroup_size, NUM_REDUCE_STAGES);
}

template <typename scalar_t, typename IdType, int64_t kVecSize>
void launchFusedQNormRopeCTAImpl(
    const scalar_t* q_ptr,
    scalar_t* out_ptr,
    const float* freqs_cis_ptr,
    const IdType* positions,
    int64_t q_token_stride,
    int64_t q_head_stride,
    int64_t out_token_stride,
    int64_t out_head_stride,
    int64_t head_dim,
    int64_t rope_dim,
    int64_t num_tokens,
    int64_t num_heads,
    float eps,
    sycl::queue& queue) {
  const int64_t num_works = num_tokens * num_heads;
  const int64_t workgroup_size = computeQNormRopeCTAWorkgroupSize(head_dim, kVecSize);

  const int64_t max_resident_wgs =
      std::max<int64_t>(1, dpcppMaxWorkItemsPerTile(dpcppGetDeviceIdOfCurrentQueue()) / workgroup_size);
  const int64_t num_wgs = std::max<int64_t>(1, std::min<int64_t>(num_works, max_resident_wgs));

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> stage(sycl::range<1>(static_cast<size_t>(head_dim)), cgh);
    FusedQNormRopeCTAKernel<scalar_t, IdType, kVecSize> kernel{
        q_ptr,
        out_ptr,
        freqs_cis_ptr,
        positions,
        q_token_stride,
        q_head_stride,
        out_token_stride,
        out_head_stride,
        head_dim,
        rope_dim,
        static_cast<uint32_t>(num_heads),
        static_cast<uint32_t>(num_tokens),
        eps,
        stage};
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(num_wgs * workgroup_size)),
            sycl::range<1>(static_cast<size_t>(workgroup_size))),
        kernel);
  });
}

template <typename scalar_t, typename IdType>
void launchFusedQNormRopeCTA(
    const scalar_t* q_ptr,
    scalar_t* out_ptr,
    const float* freqs_cis_ptr,
    const IdType* positions,
    int64_t q_token_stride,
    int64_t q_head_stride,
    int64_t out_token_stride,
    int64_t out_head_stride,
    int64_t head_dim,
    int64_t rope_dim,
    int64_t num_tokens,
    int64_t num_heads,
    float eps,
    sycl::queue& queue) {
  // `stage` in FusedQNormRopeCTAKernel is head_dim floats of SLM per
  // work-group; unlike the warp path (register-only), this scales directly
  // with head_dim, so an unbounded head_dim could request more SLM than the
  // device provides per work-group. Fail loudly instead of letting the SYCL
  // runtime reject (or silently misbehave on) an oversized local_accessor.
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const int64_t required_slm_bytes = head_dim * static_cast<int64_t>(sizeof(float));
  const int64_t available_slm_bytes = dpcppLocalMemSize(dev_id);
  TORCH_CHECK(
      required_slm_bytes <= available_slm_bytes,
      "fused_q_norm_rope (CTA path): head_dim=",
      head_dim,
      " requires ",
      required_slm_bytes,
      " bytes of SLM staging, but the device only provides ",
      available_slm_bytes,
      " bytes of local memory per work-group");

  // vec_size must evenly divide head_dim (the CTA loop strides by
  // workgroup_size * vec_size, with no OOB remainder handling).
  const int64_t vec_size = pickVecSizeForRow(
      sizeof(scalar_t), {q_ptr, out_ptr}, head_dim, {q_token_stride, q_head_stride, out_token_stride, out_head_stride});

  // head_dim is a runtime value here (unlike the warp path), so there's no
  // real compile-time elems-per-thread to pass. 16 is just the largest
  // candidate dispatchFusedQKNormRopeVecSize ever tries, used as a ceiling
  // that lets every candidate (16/8/4/2) pass its `if constexpr` filter;
  // the actual choice is made at runtime via `vec_size == kCandidate`.
  dispatchFusedQKNormRopeVecSize<16>(vec_size, [&](auto vec_size_tag) {
    constexpr int64_t kVecSize = decltype(vec_size_tag)::value;
    launchFusedQNormRopeCTAImpl<scalar_t, IdType, kVecSize>(
        q_ptr,
        out_ptr,
        freqs_cis_ptr,
        positions,
        q_token_stride,
        q_head_stride,
        out_token_stride,
        out_head_stride,
        head_dim,
        rope_dim,
        num_tokens,
        num_heads,
        eps,
        queue);
  });
}

void fused_q_norm_rope(
    torch::Tensor& q_input, torch::Tensor& q_output, torch::Tensor& freqs_cis, torch::Tensor& positions, double eps) {
  TORCH_CHECK(q_input.dim() == 3, "q_input must be 3D: [num_tokens, num_heads, head_dim]");
  TORCH_CHECK(q_output.dim() == 3, "q_output must be 3D: [num_tokens, num_heads, head_dim]");
  TORCH_CHECK(q_input.scalar_type() == q_output.scalar_type(), "q_input and q_output must have the same dtype");
  TORCH_CHECK(freqs_cis.scalar_type() == at::ScalarType::Float, "freqs_cis must be float32");
  TORCH_CHECK(freqs_cis.dim() == 2, "freqs_cis must be 2D: [max_position, rope_dim]");

  CHECK_DEVICE(q_input);
  TORCH_CHECK(q_input.stride(-1) == 1, "q_input must be contiguous in its last dimension (head_dim)");
  CHECK_DEVICE(q_output);
  TORCH_CHECK(q_output.stride(-1) == 1, "q_output must be contiguous in its last dimension (head_dim)");
  CHECK_DEVICE(freqs_cis);
  CHECK_CONTIGUOUS(freqs_cis);
  CHECK_DEVICE(positions);
  CHECK_CONTIGUOUS(positions);

  const int64_t num_tokens = q_input.size(0);
  const int64_t num_heads = q_input.size(1);
  const int64_t head_dim = q_input.size(2);
  TORCH_CHECK(
      q_output.size(0) == num_tokens && q_output.size(1) == num_heads && q_output.size(2) == head_dim,
      "q_output shape must match q_input");
  TORCH_CHECK(positions.dim() == 1, "positions must be 1D [num_tokens]");
  TORCH_CHECK(positions.size(0) == num_tokens, "positions size must match q_input's token count");

  const int64_t rope_dim = freqs_cis.size(1);
  TORCH_CHECK(rope_dim % 2 == 0, "rope_dim must be even (interleaved re/im)");
  TORCH_CHECK(rope_dim > 0 && rope_dim <= head_dim, "rope_dim must be in (0, head_dim]");

  if (num_tokens == 0) return;

  const int64_t q_token_stride = q_input.stride(0);
  const int64_t q_head_stride = q_input.stride(1);
  const int64_t out_token_stride = q_output.stride(0);
  const int64_t out_head_stride = q_output.stride(1);

  auto queue = dpcppGetCurrentQueue();

  dispatchFusedQKNormRopeScalarType<false>(q_input.scalar_type(), "fused_q_norm_rope", [&](auto scalar_tag) {
    using scalar_t = typename decltype(scalar_tag)::type;
    dispatchFusedQKNormRopePositionsType(positions.scalar_type(), "fused_q_norm_rope", [&](auto id_tag) {
      using IdType = typename decltype(id_tag)::type;

      const scalar_t* q_ptr = static_cast<const scalar_t*>(q_input.data_ptr());
      scalar_t* out_ptr = static_cast<scalar_t*>(q_output.data_ptr());
      const float* freqs_ptr = static_cast<const float*>(freqs_cis.data_ptr());
      const IdType* pos_ptr = static_cast<const IdType*>(positions.data_ptr());

      // Warp path for head_dim in {64,128,192,256} when rope_dim aligns to a
      // lane boundary; everything else (incl. head_dim > 256, e.g. DSV4's
      // 512) falls back to CTA. Try each head_dim through one lambda
      // instead of repeating the ~10-argument call three times.
      bool dispatched = false;
      auto try_warp_head_dim = [&](auto head_dim_tag) {
        constexpr int64_t kHeadDimConst = decltype(head_dim_tag)::value;
        constexpr int64_t kElemsPerThread = kHeadDimConst / NUM_REDUCE_STAGES;
        if (dispatched || head_dim != kHeadDimConst || rope_dim % kElemsPerThread != 0) return;
        dispatched = true;
        launchFusedQNormRopeWarpImpl<kHeadDimConst, scalar_t, IdType>(
            q_ptr,
            out_ptr,
            freqs_ptr,
            pos_ptr,
            q_token_stride,
            q_head_stride,
            out_token_stride,
            out_head_stride,
            rope_dim,
            num_tokens,
            num_heads,
            static_cast<float>(eps),
            queue);
      };
      try_warp_head_dim(std::integral_constant<int64_t, 64>{});
      try_warp_head_dim(std::integral_constant<int64_t, 128>{});
      try_warp_head_dim(std::integral_constant<int64_t, 192>{});
      try_warp_head_dim(std::integral_constant<int64_t, 256>{});

      if (!dispatched) {
        launchFusedQNormRopeCTA<scalar_t, IdType>(
            q_ptr,
            out_ptr,
            freqs_ptr,
            pos_ptr,
            q_token_stride,
            q_head_stride,
            out_token_stride,
            out_head_stride,
            head_dim,
            rope_dim,
            num_tokens,
            num_heads,
            static_cast<float>(eps),
            queue);
      }
    });
  });
}

// ============================================================================
// Fused K Norm + RoPE + FlashMLA Paged Cache Store
// (`fused_k_norm_rope_flashmla`)
// ============================================================================

constexpr float FP8_E4M3_MAX = 448.0f;
// Each FP8 quantization scale chunk / warp covers 64 elements
constexpr int64_t kElementsPerScaleChunk = 64;

inline uint8_t castToUE8M0(float x) {
  uint32_t bits = sycl::bit_cast<uint32_t>(x);
  uint32_t exp = (bits >> 23) & 0xFF;
  uint32_t round_up = (bits & 0x7FFFFF) != 0 ? 1 : 0;
  return static_cast<uint8_t>(exp + round_up);
}

inline float invScaleUE8M0(uint8_t ue8m0) {
  if (ue8m0 >= 254) return 0.0f;
  uint32_t inv_exp = 254 - ue8m0;
  uint32_t inv_bits = inv_exp << 23;
  return sycl::bit_cast<float>(inv_bits);
}

template <typename scalar_t, typename IdType, int64_t kVecSize>
struct FusedKNormRopeFlashMLAKernel {
  using VecT = aligned_vector_loop<scalar_t, kVecSize>;

  const scalar_t* kv_ptr;
  const scalar_t* kv_weight_ptr;
  const float* freqs_cis_ptr;
  const IdType* positions;
  const int32_t* out_loc;
  uint8_t* kvcache_ptr;
  int64_t kv_stride_batch;
  int64_t kPageBytes;
  int64_t page_size;
  int32_t page_bits;
  int64_t head_dim;
  int64_t rope_dim;
  uint32_t num_tokens;
  float eps;

  sycl::local_accessor<float, 1> stage;    // [head_dim]
  sycl::local_accessor<float, 1> stage_w;  // [head_dim]

  [[sycl::reqd_sub_group_size(NUM_REDUCE_STAGES)]] void operator()(sycl::nd_item<1> item) const {
    const int64_t local_id = static_cast<int64_t>(item.get_local_id(0));
    const int64_t workgroup_size = static_cast<int64_t>(item.get_local_range(0));
    const int64_t workgroup_id = static_cast<int64_t>(item.get_group(0));
    const int64_t num_workgroups = static_cast<int64_t>(item.get_group_range(0));
    const int64_t sg_id = local_id / NUM_REDUCE_STAGES;
    const int64_t lane_id = local_id % NUM_REDUCE_STAGES;
    const int64_t nope_dim = head_dim - rope_dim;
    const int64_t num_nope_sgs = divUp<int64_t>(nope_dim, kElementsPerScaleChunk);
    const int64_t value_bytes = nope_dim + rope_dim * 2;
    const int64_t scale_slot_bytes = divUp<int64_t>(num_nope_sgs, 8) * 8;

    for (int64_t token_id = workgroup_id; token_id < static_cast<int64_t>(num_tokens); token_id += num_workgroups) {
      const int32_t slot = out_loc[token_id];

      // Phase 1: load kv and kv_weight into SLM, compute sum of squares
      const scalar_t* row_kv = kv_ptr + token_id * kv_stride_batch;
      float sum_of_squares = 0.0f;

      for (int64_t dim = local_id * kVecSize; dim < head_dim; dim += workgroup_size * kVecSize) {
        const VecT in_vec = *reinterpret_cast<const VecT*>(row_kv + dim);
        const VecT w_vec = *reinterpret_cast<const VecT*>(kv_weight_ptr + dim);
#pragma unroll
        for (int64_t v = 0; v < kVecSize; ++v) {
          const float x = static_cast<float>(in_vec[v]);
          const float w = static_cast<float>(w_vec[v]);
          stage[dim + v] = x;
          stage_w[dim + v] = w;
          sum_of_squares += x * x;
        }
      }

      sum_of_squares = sycl::reduce_over_group(item.get_group(), sum_of_squares, sycl::plus<float>());
      const float norm_factor = sycl::rsqrt(sum_of_squares / static_cast<float>(head_dim) + eps);

      item.barrier(sycl::access::fence_space::local_space);

      // Phase 2: normalize values in SLM
      for (int64_t dim = local_id * kVecSize; dim < head_dim; dim += workgroup_size * kVecSize) {
#pragma unroll
        for (int64_t v = 0; v < kVecSize; ++v) {
          const int64_t d = dim + v;
          stage[d] = stage[d] * norm_factor * stage_w[d];
        }
      }

      item.barrier(sycl::access::fence_space::local_space);

      // If slot < 0, skip cache write (out of window or padded token)
      if (slot < 0) {
        item.barrier(sycl::access::fence_space::local_space);
        continue;
      }

      const int64_t page = slot >> page_bits;
      const int64_t offset = slot & (page_size - 1);
      uint8_t* page_ptr = kvcache_ptr + page * kPageBytes;
      uint8_t* value_ptr = page_ptr + offset * value_bytes;
      uint8_t* scale_ptr = page_ptr + page_size * value_bytes + offset * scale_slot_bytes;

      // Phase 3: NoPE FP8 quant (for sub-groups 0..num_nope_sgs-1) and RoPE (for remaining sub-groups)
      if (sg_id < num_nope_sgs) {
        // Each sub-group (16 lanes) handles 1 chunk of 64 elements (4 elems/lane)
        const int64_t chunk_idx = sg_id;
        const int64_t chunk_start = chunk_idx * kElementsPerScaleChunk;
        float local_max = 0.0f;
        float x_vals[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
        for (int64_t v = 0; v < 4; ++v) {
          const int64_t d = chunk_start + lane_id * 4 + v;
          if (d < nope_dim) {
            const float x = stage[d];
            x_vals[v] = x;
            local_max = sycl::fmax(local_max, sycl::fabs(x));
          }
        }

        const float abs_max = sycl::reduce_over_group(item.get_sub_group(), local_max, sycl::maximum<float>());
        const float scale_raw = sycl::fmax(1e-4f, abs_max) / FP8_E4M3_MAX;
        const uint8_t ue8m0 = castToUE8M0(scale_raw);

        if (lane_id == 0) {
          scale_ptr[chunk_idx] = ue8m0;
        }

        const float inv_scale = invScaleUE8M0(ue8m0);

#pragma unroll
        for (int64_t v = 0; v < 4; ++v) {
          const int64_t d = chunk_start + lane_id * 4 + v;
          if (d < nope_dim) {
            float qval = x_vals[v] * inv_scale;
            qval = sycl::fmax(-FP8_E4M3_MAX, sycl::fmin(qval, FP8_E4M3_MAX));
            const cutlass::float_e4m3_t fp8_elem = static_cast<cutlass::float_e4m3_t>(qval);
            value_ptr[d] = sycl::bit_cast<uint8_t>(fp8_elem);
          }
        }
      } else {
        // RoPE sub-groups: handle elements from nope_dim to head_dim
        const int64_t pos = static_cast<int64_t>(positions[token_id]);
        const float* freq_row = freqs_cis_ptr + pos * rope_dim;
        const int64_t rope_sg_idx = sg_id - num_nope_sgs;
        const int64_t num_rope_sgs = divUp<int64_t>(rope_dim, kElementsPerScaleChunk);

        if (rope_sg_idx < num_rope_sgs) {
          const int64_t rope_chunk_start = rope_sg_idx * kElementsPerScaleChunk;
          // Each sub-group (16 lanes) handles 32 complex pairs (2 pairs = 4 elems / lane)
#pragma unroll
          for (int64_t pair_step = 0; pair_step < 2; ++pair_step) {
            const int64_t pair_idx = rope_chunk_start / 2 + lane_id * 2 + pair_step;
            const int64_t p = pair_idx * 2;  // relative index in rope_dim
            if (p + 1 < rope_dim) {
              const int64_t abs_d = nope_dim + p;
              const float x_re = stage[abs_d];
              const float x_im = stage[abs_d + 1];
              const float f_re = freq_row[p];
              const float f_im = freq_row[p + 1];

              const float rot_re = x_re * f_re - x_im * f_im;
              const float rot_im = x_re * f_im + x_im * f_re;

              const sycl::ext::oneapi::bfloat16 bf_re(rot_re);
              const sycl::ext::oneapi::bfloat16 bf_im(rot_im);

              auto* rope_dest = reinterpret_cast<sycl::ext::oneapi::bfloat16*>(value_ptr + nope_dim) + p;
              rope_dest[0] = bf_re;
              rope_dest[1] = bf_im;
            }
          }
        }

        // Fill padding bytes in scale slot if last sub-group lane 0
        if (sg_id == (workgroup_size / NUM_REDUCE_STAGES - 1) && lane_id == 0) {
          for (int64_t pad = num_nope_sgs; pad < scale_slot_bytes; ++pad) {
            scale_ptr[pad] = 0;
          }
        }
      }

      item.barrier(sycl::access::fence_space::local_space);
    }
  }
};

template <typename scalar_t, typename IdType, int64_t kVecSize>
void launchFusedKNormRopeFlashMLAImpl(
    const scalar_t* kv_ptr,
    const scalar_t* kv_weight_ptr,
    const float* freqs_cis_ptr,
    const IdType* positions,
    const int32_t* out_loc,
    uint8_t* kvcache_ptr,
    int64_t kv_stride_batch,
    int64_t kPageBytes,
    int64_t page_size,
    int32_t page_bits,
    int64_t head_dim,
    int64_t rope_dim,
    int64_t num_tokens,
    float eps,
    sycl::queue& queue) {
  const int64_t nope_dim = head_dim - rope_dim;
  const int64_t num_sgs =
      divUp<int64_t>(nope_dim, kElementsPerScaleChunk) + divUp<int64_t>(rope_dim, kElementsPerScaleChunk);
  const int64_t workgroup_size = std::max<int64_t>(NUM_REDUCE_STAGES, num_sgs * NUM_REDUCE_STAGES);

  const int64_t max_resident_wgs =
      std::max<int64_t>(1, dpcppMaxWorkItemsPerTile(dpcppGetDeviceIdOfCurrentQueue()) / workgroup_size);
  const int64_t num_wgs = std::max<int64_t>(1, std::min<int64_t>(num_tokens, max_resident_wgs));

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> stage(sycl::range<1>(static_cast<size_t>(head_dim)), cgh);
    sycl::local_accessor<float, 1> stage_w(sycl::range<1>(static_cast<size_t>(head_dim)), cgh);

    FusedKNormRopeFlashMLAKernel<scalar_t, IdType, kVecSize> kernel{
        kv_ptr,
        kv_weight_ptr,
        freqs_cis_ptr,
        positions,
        out_loc,
        kvcache_ptr,
        kv_stride_batch,
        kPageBytes,
        page_size,
        page_bits,
        head_dim,
        rope_dim,
        static_cast<uint32_t>(num_tokens),
        eps,
        stage,
        stage_w};

    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<size_t>(num_wgs * workgroup_size)),
            sycl::range<1>(static_cast<size_t>(workgroup_size))),
        kernel);
  });
}

template <typename scalar_t, typename IdType>
void launchFusedKNormRopeFlashMLA(
    const scalar_t* kv_ptr,
    const scalar_t* kv_weight_ptr,
    const float* freqs_cis_ptr,
    const IdType* positions,
    const int32_t* out_loc,
    uint8_t* kvcache_ptr,
    int64_t kv_stride_batch,
    int64_t kPageBytes,
    int64_t page_size,
    int32_t page_bits,
    int64_t head_dim,
    int64_t rope_dim,
    int64_t num_tokens,
    float eps,
    sycl::queue& queue) {
  const auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const int64_t required_slm_bytes = 2 * head_dim * static_cast<int64_t>(sizeof(float));
  const int64_t available_slm_bytes = dpcppLocalMemSize(dev_id);
  TORCH_CHECK(
      required_slm_bytes <= available_slm_bytes,
      "fused_k_norm_rope_flashmla: head_dim=",
      head_dim,
      " requires ",
      required_slm_bytes,
      " bytes of SLM staging, but the device only provides ",
      available_slm_bytes,
      " bytes of local memory per work-group");

  const int64_t vec_size = pickVecSizeForRow(sizeof(scalar_t), {kv_ptr, kv_weight_ptr}, head_dim, {kv_stride_batch});

  dispatchFusedQKNormRopeVecSize<16>(vec_size, [&](auto vec_size_tag) {
    constexpr int64_t kVecSize = decltype(vec_size_tag)::value;
    launchFusedKNormRopeFlashMLAImpl<scalar_t, IdType, kVecSize>(
        kv_ptr,
        kv_weight_ptr,
        freqs_cis_ptr,
        positions,
        out_loc,
        kvcache_ptr,
        kv_stride_batch,
        kPageBytes,
        page_size,
        page_bits,
        head_dim,
        rope_dim,
        num_tokens,
        eps,
        queue);
  });
}

void fused_k_norm_rope_flashmla(
    torch::Tensor& kv,
    torch::Tensor& kv_weight,
    torch::Tensor& freqs_cis,
    torch::Tensor& positions,
    torch::Tensor& out_loc,
    torch::Tensor& kvcache,
    double eps,
    int64_t page_size) {
  TORCH_CHECK(kv.dim() == 2, "kv must be 2D [num_tokens, head_dim]");
  const int64_t head_dim = kv.size(1);
  TORCH_CHECK(kv_weight.dim() == 1, "kv_weight must be 1D [head_dim]");
  TORCH_CHECK(
      kv_weight.size(0) == head_dim, "kv_weight size (", kv_weight.size(0), ") must match head_dim (", head_dim, ")");
  TORCH_CHECK(kv.scalar_type() == kv_weight.scalar_type(), "kv and kv_weight must have the same dtype");
  TORCH_CHECK(freqs_cis.scalar_type() == at::ScalarType::Float, "freqs_cis must be float32");
  TORCH_CHECK(freqs_cis.dim() == 2, "freqs_cis must be 2D [max_pos, rope_dim]");
  const int64_t rope_dim = freqs_cis.size(1);
  TORCH_CHECK(rope_dim > 0 && rope_dim <= head_dim, "rope_dim must be in (0, head_dim]");
  TORCH_CHECK(positions.dim() == 1, "positions must be 1D [num_tokens]");
  TORCH_CHECK(out_loc.dim() == 1, "out_loc must be 1D [num_tokens]");
  TORCH_CHECK(out_loc.scalar_type() == at::ScalarType::Int, "out_loc must be int32");
  TORCH_CHECK(kvcache.dim() == 2, "kvcache must be 2D [npages, kPageBytes]");
  TORCH_CHECK(kvcache.scalar_type() == at::ScalarType::Byte, "kvcache must be uint8");

  CHECK_DEVICE(kv);
  TORCH_CHECK(kv.stride(-1) == 1, "kv must be contiguous in its last dimension (head_dim)");
  CHECK_DEVICE(kv_weight);
  CHECK_CONTIGUOUS(kv_weight);
  CHECK_DEVICE(freqs_cis);
  CHECK_CONTIGUOUS(freqs_cis);
  CHECK_DEVICE(positions);
  CHECK_CONTIGUOUS(positions);
  CHECK_DEVICE(out_loc);
  CHECK_CONTIGUOUS(out_loc);
  CHECK_DEVICE(kvcache);
  TORCH_CHECK(kvcache.stride(-1) == 1, "kvcache must be contiguous in its last dimension");

  const int64_t num_tokens = kv.size(0);
  TORCH_CHECK(positions.size(0) == num_tokens, "positions size must match kv token count");
  TORCH_CHECK(out_loc.size(0) == num_tokens, "out_loc size must match kv token count");

  TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0, "page_size must be a power of 2, got ", page_size);

  int32_t page_bits = 0;
  while ((1LL << page_bits) < page_size) {
    page_bits++;
  }

  const int64_t kPageBytes = kvcache.stride(0);
  const int64_t nope_dim = head_dim - rope_dim;
  const int64_t value_bytes = nope_dim + rope_dim * 2;
  const int64_t scale_slot_bytes = divUp<int64_t>(divUp<int64_t>(nope_dim, kElementsPerScaleChunk), 8) * 8;
  const int64_t min_page_bytes = page_size * (value_bytes + scale_slot_bytes);
  TORCH_CHECK(
      kPageBytes >= min_page_bytes,
      "kvcache stride(0) (",
      kPageBytes,
      ") must be at least page_size * ",
      value_bytes + scale_slot_bytes,
      " (",
      min_page_bytes,
      ")");

  if (num_tokens == 0) return;

  const int64_t kv_stride_batch = kv.stride(0);
  auto queue = dpcppGetCurrentQueue();

  dispatchFusedQKNormRopeScalarType<false>(kv.scalar_type(), "fused_k_norm_rope_flashmla", [&](auto scalar_tag) {
    using scalar_t = typename decltype(scalar_tag)::type;
    dispatchFusedQKNormRopePositionsType(positions.scalar_type(), "fused_k_norm_rope_flashmla", [&](auto id_tag) {
      using IdType = typename decltype(id_tag)::type;

      const scalar_t* kv_ptr = static_cast<const scalar_t*>(kv.data_ptr());
      const scalar_t* kv_weight_ptr = static_cast<const scalar_t*>(kv_weight.data_ptr());
      const float* freqs_ptr = static_cast<const float*>(freqs_cis.data_ptr());
      const IdType* pos_ptr = static_cast<const IdType*>(positions.data_ptr());
      const int32_t* out_loc_ptr = static_cast<const int32_t*>(out_loc.data_ptr());
      uint8_t* kvcache_ptr = static_cast<uint8_t*>(kvcache.data_ptr());

      launchFusedKNormRopeFlashMLA<scalar_t, IdType>(
          kv_ptr,
          kv_weight_ptr,
          freqs_ptr,
          pos_ptr,
          out_loc_ptr,
          kvcache_ptr,
          kv_stride_batch,
          kPageBytes,
          page_size,
          page_bits,
          head_dim,
          rope_dim,
          num_tokens,
          static_cast<float>(eps),
          queue);
    });
  });
}

void fused_inplace_qknorm_rope(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    torch::Tensor& cos_sin_cache,
    torch::Tensor& positions,
    bool is_neox,
    double eps,
    int64_t head_dim,
    int64_t rope_dim) {
  TORCH_CHECK(q.dim() == k.dim(), "q and k must have the same rank, got q:", q.dim(), " k:", k.dim());
  TORCH_CHECK(q.dim() == 3 || q.dim() == 4, "q/k must be 3D or 4D tensors, got q:", q.dim());
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must have the same dtype");
  TORCH_CHECK(q_weight.scalar_type() == q.scalar_type(), "q_weight dtype must match q dtype");
  TORCH_CHECK(k_weight.scalar_type() == k.scalar_type(), "k_weight dtype must match k dtype");
  TORCH_CHECK(cos_sin_cache.scalar_type() == at::ScalarType::Float, "cos_sin_cache must be float32");

  CHECK_DEVICE(q);
  // Only the last dimension (head_dim) needs to be contiguous; q/k may be
  // views sliced out of a larger packed buffer (e.g. per-head strides that
  // don't equal head_dim), so full tensor contiguity is not required. Token
  // and head strides are read directly below instead of being assumed.
  TORCH_CHECK(q.stride(-1) == 1, "q must be contiguous in its last dimension (head_dim)");
  CHECK_DEVICE(k);
  TORCH_CHECK(k.stride(-1) == 1, "k must be contiguous in its last dimension (head_dim)");
  CHECK_DEVICE(q_weight);
  CHECK_CONTIGUOUS(q_weight);
  CHECK_DEVICE(k_weight);
  CHECK_CONTIGUOUS(k_weight);
  CHECK_DEVICE(cos_sin_cache);
  CHECK_CONTIGUOUS(cos_sin_cache);
  CHECK_DEVICE(positions);
  CHECK_CONTIGUOUS(positions);

  if (q.dim() == 4) {
    TORCH_CHECK(
        q.stride(0) == q.size(1) * q.stride(1),
        "q batch and sequence dimensions must be mergeable (i.e. contiguous with each other) for 4D input");
  }
  if (k.dim() == 4) {
    TORCH_CHECK(
        k.stride(0) == k.size(1) * k.stride(1),
        "k batch and sequence dimensions must be mergeable (i.e. contiguous with each other) for 4D input");
  }

  auto q_view = q.dim() == 4 ? q.view({-1, q.size(2), q.size(3)}) : q;
  auto k_view = k.dim() == 4 ? k.view({-1, k.size(2), k.size(3)}) : k;
  TORCH_CHECK(q_view.dim() == 3 && k_view.dim() == 3, "Flattened q/k must be 3D tensors");
  TORCH_CHECK(q_view.size(0) == k_view.size(0), "q and k must have the same token count after flattening");
  TORCH_CHECK(q_view.size(2) == k_view.size(2), "q and k must have the same head_dim");

  const int64_t num_tokens = q_view.size(0);
  const int64_t num_qo_heads = q_view.size(1);
  const int64_t num_kv_heads = k_view.size(1);
  const int64_t inferred_head_dim = q_view.size(2);
  const int64_t q_head_stride = q_view.stride(1);
  const int64_t k_head_stride = k_view.stride(1);

  TORCH_CHECK(q_weight.dim() == 1, "q_weight must be 1D [head_dim]");
  TORCH_CHECK(k_weight.dim() == 1, "k_weight must be 1D [head_dim]");
  TORCH_CHECK(q_weight.size(0) == inferred_head_dim, "q_weight size must match head_dim");
  TORCH_CHECK(k_weight.size(0) == inferred_head_dim, "k_weight size must match head_dim");
  TORCH_CHECK(cos_sin_cache.dim() == 2, "cos_sin_cache must be 2D [max_position, rope_dim]");
  const int64_t inferred_rope_dim = cos_sin_cache.size(1);
  if (head_dim != 0) {
    TORCH_CHECK(
        head_dim == inferred_head_dim,
        "head_dim must match q/k hidden size, got ",
        head_dim,
        " vs ",
        inferred_head_dim);
  }
  if (rope_dim != 0) {
    TORCH_CHECK(
        rope_dim == inferred_rope_dim,
        "rope_dim must match cos_sin_cache width, got ",
        rope_dim,
        " vs ",
        inferred_rope_dim);
  }
  head_dim = inferred_head_dim;
  rope_dim = inferred_rope_dim;
  TORCH_CHECK(rope_dim % 2 == 0, "rope_dim must be even");
  TORCH_CHECK(rope_dim <= head_dim, "rope_dim must be <= head_dim");
  TORCH_CHECK(positions.dim() == 1, "positions must be 1D [num_tokens]");
  TORCH_CHECK(positions.size(0) == num_tokens, "positions size must match flattened q/k tokens");

  auto queue = dpcppGetCurrentQueue();

  dispatchFusedQKNormRopeScalarType<false>(q_view.scalar_type(), "fused_inplace_qknorm_rope", [&](auto scalar_tag) {
    using scalar_t = typename decltype(scalar_tag)::type;
    dispatchFusedQKNormRopePositionsType(positions.scalar_type(), "fused_inplace_qknorm_rope", [&](auto id_tag) {
      using IdType = typename decltype(id_tag)::type;
      dispatchFusedQKNormRopeHeadDim(head_dim, "fused_inplace_qknorm_rope", [&](auto head_dim_tag) {
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
              q_head_stride,
              k_head_stride,
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
              q_head_stride,
              k_head_stride,
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
