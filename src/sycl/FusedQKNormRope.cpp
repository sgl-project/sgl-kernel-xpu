#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#include "MemoryAccess.h"
#include "Norm.h"
#include "SYCLHelpers.h"
#include "Utils.h"

namespace at::native::xpu {

template <typename T>
inline T divUp(T m, T n) {
  return (m + n - 1) / n;
}

// Sub-group reduction for sum
template <typename T>
inline T subGroupReduceSum(T val, const sycl::sub_group& sg) {
  for (int offset = sg.get_max_local_range()[0] / 2; offset > 0; offset /= 2) {
    val += sycl::shift_group_left(sg, val, offset);
  }
  return val;
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

// SYCL Kernel for Fused QK Norm and RoPE
template <int head_dim, bool interleave, typename scalar_t>
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

  void operator()(sycl::nd_item<1> item) const {
    using accscalar_t = at::opmath_type<scalar_t>;

    const int sg_size = item.get_sub_group().get_max_local_range()[0];
    const int warpsPerBlock = item.get_local_range(0) / sg_size;
    const int warpId = item.get_local_id(0) / sg_size;
    const int laneId = item.get_local_id(0) % sg_size;

    const int globalWarpIdx = item.get_group(0) * warpsPerBlock + warpId;
    const int total_qk_heads = num_heads_q + num_heads_k;

    const int tokenIdx = globalWarpIdx / total_qk_heads;
    const int localHeadIdx = globalWarpIdx % total_qk_heads;

    if (tokenIdx >= num_tokens) return;

    const bool isQ = localHeadIdx < num_heads_q;
    const int headIdx = isQ ? localHeadIdx : localHeadIdx - num_heads_q;
    const int num_heads = num_heads_q + num_heads_k + num_heads_v;

    constexpr int numElemsPerThread = head_dim / 32;
    accscalar_t elements[numElemsPerThread];

    int offsetWarp;
    if (isQ) {
      offsetWarp = tokenIdx * num_heads * head_dim + headIdx * head_dim;
    } else {
      offsetWarp = tokenIdx * num_heads * head_dim + num_heads_q * head_dim + headIdx * head_dim;
    }
    int offsetThread = offsetWarp + laneId * numElemsPerThread;

    // Load data and compute sum of squares for RMSNorm
    accscalar_t sumOfSquares = 0.0f;
    for (int i = 0; i < numElemsPerThread; i++) {
      elements[i] = static_cast<accscalar_t>(qkv[offsetThread + i]);
      sumOfSquares += elements[i] * elements[i];
    }

    // Reduce sum across sub-group (warp)
    auto sg = item.get_sub_group();
    sumOfSquares = sycl::reduce_over_group(sg, sumOfSquares, sycl::plus<accscalar_t>());

    // Compute RMS normalization factor
    accscalar_t rms_rcp =
        sycl::rsqrt(sumOfSquares / static_cast<accscalar_t>(head_dim) + static_cast<accscalar_t>(eps));

    // Normalize elements
    for (int i = 0; i < numElemsPerThread; i++) {
      int dim = laneId * numElemsPerThread + i;
      accscalar_t weight = isQ ? static_cast<accscalar_t>(q_weight[dim]) : static_cast<accscalar_t>(k_weight[dim]);
      elements[i] *= rms_rcp * weight;
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
    for (int i = 0; i < numElemsPerThread; i++) {
      qkv[offsetThread + i] = static_cast<scalar_t>(elements[i]);
    }
  }
};
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
  constexpr int blockSize = 256;
  const int warpsPerBlock = blockSize / 32;
  const int totalQKHeads = num_heads_q + num_heads_k;
  const int totalWarps = num_tokens * totalQKHeads;
  const int gridSize = divUp(totalWarps, warpsPerBlock);

  FusedQKNormRopeKernel<head_dim, interleave, scalar_t> kernel{
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
  TORCH_CHECK(rotary_dim % (head_dim / 32) == 0, "rotary_dim must be divisible by numElemsPerThread");

  if (is_neox) {
    int64_t half_rotary_lanes = rotary_dim / (head_dim / 32) / 2;
    TORCH_CHECK(
        half_rotary_lanes >= 1 && (half_rotary_lanes & (half_rotary_lanes - 1)) == 0,
        "half_rotary_lanes must be a power of 2 for neox style, got ",
        half_rotary_lanes);
  }

  CHECK_INPUT(qkv);
  CHECK_INPUT(position_ids);
  CHECK_INPUT(q_weight);
  CHECK_INPUT(k_weight);

  int64_t num_tokens = qkv.size(0);
  TORCH_CHECK(position_ids.size(0) == num_tokens, "Number of tokens in position_ids must match QKV");

  int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
  TORCH_CHECK(
      qkv.size(1) == total_heads * head_dim, "QKV tensor size must match total number of heads and head dimension");

  auto queue = dpcppGetCurrentQueue();
  bool interleave = !is_neox;

#define LAUNCH_KERNEL(head_dim, interleave)                                                          \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                                                   \
      at::ScalarType::Half, at::ScalarType::BFloat16, qkv.scalar_type(), "fused_qk_norm_rope", [&] { \
        launchFusedQKNormRopeImpl<head_dim, interleave, scalar_t>(                                   \
            qkv.data_ptr(),                                                                          \
            static_cast<int>(num_tokens),                                                            \
            static_cast<int>(num_heads_q),                                                           \
            static_cast<int>(num_heads_k),                                                           \
            static_cast<int>(num_heads_v),                                                           \
            static_cast<float>(eps),                                                                 \
            q_weight.data_ptr(),                                                                     \
            k_weight.data_ptr(),                                                                     \
            static_cast<float>(base),                                                                \
            position_ids.data_ptr<int>(),                                                            \
            static_cast<float>(factor),                                                              \
            static_cast<float>(low),                                                                 \
            static_cast<float>(high),                                                                \
            static_cast<float>(attention_factor),                                                    \
            static_cast<int>(rotary_dim),                                                            \
            queue);                                                                                  \
      });

  switch (head_dim) {
    case 64:
      if (interleave) {
        LAUNCH_KERNEL(64, true);
      } else {
        LAUNCH_KERNEL(64, false);
      }
      break;
    case 128:
      if (interleave) {
        LAUNCH_KERNEL(128, true);
      } else {
        LAUNCH_KERNEL(128, false);
      }
      break;
    case 256:
      if (interleave) {
        LAUNCH_KERNEL(256, true);
      } else {
        LAUNCH_KERNEL(256, false);
      }
      break;
    default:
      TORCH_CHECK(false, "Unsupported head dimension for fusedQKNormRope: ", head_dim);
  }

#undef LAUNCH_KERNEL
}

}  // namespace at::native::xpu
