#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>
#include <cassert>

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
  static_assert(std::is_integral<T>::value, "divUp requires an integral type");
  return (m + n - 1) / n;
}

// local to avoid ODR issues while keeping the implementation consistent.
static inline float
compute_freq_yarn_rope(float base, int rotary_dim, int half_dim, float factor, float low, float high) {
  float freq = sycl::pow(base, -2.0f * static_cast<float>(half_dim) / static_cast<float>(rotary_dim));

  if (factor != 1.0f) {
    float inv_freq_extrapolation = freq;
    float inv_freq_interpolation = freq / factor;

    float high_adj = high;
    if (sycl::fabs(low - high_adj) <= 1e-6f) {
      high_adj += 0.001f;
    }

    float dim_value = 2.0f * static_cast<float>(half_dim);
    float linear_func = (dim_value - low) / (high_adj - low);
    float ramp_func = sycl::fmin(sycl::fmax(linear_func, 0.0f), 1.0f);

    freq = inv_freq_interpolation * (1.0f - ramp_func) + inv_freq_extrapolation * ramp_func;
  }

  return freq;
}

// Applies per-dimension weight scaling followed by RoPE to Q and K heads.
template <int head_dim, bool interleave, typename scalar_t>
struct FusedQKRopeKernel {
  scalar_t* qkv;
  int num_heads_q;
  int num_heads_k;
  int num_heads_v;
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

  [[sycl::reqd_sub_group_size(32)]]
  void operator()(sycl::nd_item<1> item) const {
    using accscalar_t = float;

    const int sg_size = item.get_sub_group().get_max_local_range()[0];
    assert(sg_size == 32);
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

    // Load elements and apply per-dimension weight scaling (no RMSNorm)
    const scalar_t* weight_ptr = isQ ? q_weight : k_weight;
    for (int i = 0; i < numElemsPerThread; i++) {
      // TODO: For FP8, per-tensor or per-channel dequant scales should be applied
      // here before the weight multiply (e.g. val *= dequant_scale).
      // Currently operating at unit scale; callers must pre-scale if needed.
      const int dim = laneId * numElemsPerThread + i;
      const accscalar_t val = static_cast<accscalar_t>(qkv[offsetThread + i]);
      elements[i] = val * static_cast<accscalar_t>(weight_ptr[dim]);
    }

    // Apply RoPE to weighted elements
    accscalar_t elements2[numElemsPerThread];
    accscalar_t cos_vals[numElemsPerThread];
    accscalar_t sin_vals[numElemsPerThread];
    float pos_id = static_cast<float>(position_ids[tokenIdx]);
    const int rotary_lanes = rotary_dim / numElemsPerThread;
    const bool applyRotary = (laneId < rotary_lanes);

    auto sg = item.get_sub_group();

    if (applyRotary) {
      if constexpr (interleave) {
        // Interleave mode
        for (int i = 0; i < numElemsPerThread; i++) {
          elements2[i] = (i % 2 == 0) ? -elements[i + 1] : elements[i - 1];

          int dim_idx = laneId * numElemsPerThread + i;
          int half_dim = dim_idx / 2;
          float freq = compute_freq_yarn_rope(base, rotary_dim, half_dim, factor, low, high);
          float theta = pos_id * freq;
          sin_vals[i] = sycl::sin(theta);
          cos_vals[i] = sycl::cos(theta);
        }
      }
    }

    if constexpr (!interleave) {
      // Neox style
      sycl::group_barrier(sg);
      const int half_rotary_lanes = rotary_lanes / 2;

      for (int i = 0; i < numElemsPerThread; i++) {
        auto permuted = sycl::permute_group_by_xor(sg, elements[i], half_rotary_lanes);

        if (applyRotary) {
          elements2[i] = permuted;
          if (laneId < half_rotary_lanes) {
            elements2[i] = -elements2[i];
          }

          int dim_idx = laneId * numElemsPerThread + i;
          dim_idx = (dim_idx * 2) % rotary_dim;
          int half_dim = dim_idx / 2;
          float freq = compute_freq_yarn_rope(base, rotary_dim, half_dim, factor, low, high);
          float theta = pos_id * freq;
          sin_vals[i] = sycl::sin(theta);
          cos_vals[i] = sycl::cos(theta);
        }
      }
      sycl::group_barrier(sg);
    }

    // Apply rotation with attention_factor
    if (applyRotary) {
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
void launchFusedQKRopeImpl(
    void* qkv,
    int num_tokens,
    int num_heads_q,
    int num_heads_k,
    int num_heads_v,
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

  FusedQKRopeKernel<head_dim, interleave, scalar_t> kernel{
      static_cast<scalar_t*>(qkv),
      num_heads_q,
      num_heads_k,
      num_heads_v,
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

void fused_qk_rope(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
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
  TORCH_CHECK(
      q_weight.scalar_type() == qkv.scalar_type(),
      "Query weights dtype must match QKV dtype: expected ",
      qkv.scalar_type(),
      " but got ",
      q_weight.scalar_type());
  TORCH_CHECK(
      k_weight.scalar_type() == qkv.scalar_type(),
      "Key weights dtype must match QKV dtype: expected ",
      qkv.scalar_type(),
      " but got ",
      k_weight.scalar_type());
  TORCH_CHECK(
      position_ids.scalar_type() == at::kInt,
      "Position IDs dtype must be int32 (at::kInt) to match data_ptr<int>() usage; got ",
      position_ids.scalar_type());
  TORCH_CHECK(head_dim >= 32, "head_dim must be >= 32 to avoid invalid rotary configuration; got ", head_dim);
  TORCH_CHECK(
      rotary_dim > 0 && rotary_dim <= head_dim,
      "rotary_dim must be in the range (0, head_dim], got ",
      rotary_dim,
      " with head_dim ",
      head_dim);
  TORCH_CHECK(rotary_dim % 2 == 0, "rotary_dim must be even for RoPE, got ", rotary_dim);
  TORCH_CHECK(rotary_dim % (head_dim / 32) == 0, "rotary_dim must be divisible by numElemsPerThread");

  if (!is_neox) {
    // interleave mode: nothing extra to check
  } else {
    int64_t half_rotary_lanes = rotary_dim / (head_dim / 32) / 2;
    TORCH_CHECK(
        half_rotary_lanes >= 1 && half_rotary_lanes < 32 &&
            (half_rotary_lanes & (half_rotary_lanes - 1)) == 0,
        "half_rotary_lanes must be a power of 2 and less than 32 for neox style, got ",
        half_rotary_lanes);
  }

  CHECK_INPUT(qkv);
  CHECK_INPUT(position_ids);
  CHECK_INPUT(q_weight);
  CHECK_INPUT(k_weight);

  TORCH_CHECK(
      position_ids.scalar_type() == at::kInt,
      "position_ids must be an int32 tensor, but got ",
      position_ids.scalar_type());

  int64_t num_tokens = qkv.size(0);
  TORCH_CHECK(position_ids.size(0) == num_tokens, "Number of tokens in position_ids must match QKV");

  int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
  TORCH_CHECK(
      qkv.size(1) == total_heads * head_dim, "QKV tensor size must match total number of heads and head dimension");

  auto queue = dpcppGetCurrentQueue();
  bool interleave = !is_neox;

#define LAUNCH_QK_ROPE_KERNEL(head_dim, interleave)                             \
  do {                                                                          \
    auto dtype = qkv.scalar_type();                                             \
    if (dtype == at::ScalarType::Half) {                                        \
      launchFusedQKRopeImpl<head_dim, interleave, sycl::half>(                  \
          qkv.data_ptr(),                                                       \
          static_cast<int>(num_tokens),                                         \
          static_cast<int>(num_heads_q),                                        \
          static_cast<int>(num_heads_k),                                        \
          static_cast<int>(num_heads_v),                                        \
          q_weight.data_ptr(),                                                  \
          k_weight.data_ptr(),                                                  \
          static_cast<float>(base),                                             \
          position_ids.data_ptr<int>(),                                         \
          static_cast<float>(factor),                                           \
          static_cast<float>(low),                                              \
          static_cast<float>(high),                                             \
          static_cast<float>(attention_factor),                                 \
          static_cast<int>(rotary_dim),                                         \
          queue);                                                               \
    } else if (dtype == at::ScalarType::BFloat16) {                             \
      launchFusedQKRopeImpl<head_dim, interleave, sycl::ext::oneapi::bfloat16>( \
          qkv.data_ptr(),                                                       \
          static_cast<int>(num_tokens),                                         \
          static_cast<int>(num_heads_q),                                        \
          static_cast<int>(num_heads_k),                                        \
          static_cast<int>(num_heads_v),                                        \
          q_weight.data_ptr(),                                                  \
          k_weight.data_ptr(),                                                  \
          static_cast<float>(base),                                             \
          position_ids.data_ptr<int>(),                                         \
          static_cast<float>(factor),                                           \
          static_cast<float>(low),                                              \
          static_cast<float>(high),                                             \
          static_cast<float>(attention_factor),                                 \
          static_cast<int>(rotary_dim),                                         \
          queue);                                                               \
    } else if (dtype == at::ScalarType::Float8_e4m3fn) {                        \
      launchFusedQKRopeImpl<head_dim, interleave, float_e4m3_t>(                \
          qkv.data_ptr(),                                                       \
          static_cast<int>(num_tokens),                                         \
          static_cast<int>(num_heads_q),                                        \
          static_cast<int>(num_heads_k),                                        \
          static_cast<int>(num_heads_v),                                        \
          q_weight.data_ptr(),                                                  \
          k_weight.data_ptr(),                                                  \
          static_cast<float>(base),                                             \
          position_ids.data_ptr<int>(),                                         \
          static_cast<float>(factor),                                           \
          static_cast<float>(low),                                              \
          static_cast<float>(high),                                             \
          static_cast<float>(attention_factor),                                 \
          static_cast<int>(rotary_dim),                                         \
          queue);                                                               \
    } else {                                                                    \
      TORCH_CHECK(false, "Unsupported dtype for fused_qk_rope: ", dtype);       \
    }                                                                           \
  } while (0)

  switch (head_dim) {
    case 64:
      if (interleave) {
        LAUNCH_QK_ROPE_KERNEL(64, true);
      } else {
        LAUNCH_QK_ROPE_KERNEL(64, false);
      }
      break;
    case 128:
      if (interleave) {
        LAUNCH_QK_ROPE_KERNEL(128, true);
      } else {
        LAUNCH_QK_ROPE_KERNEL(128, false);
      }
      break;
    case 256:
      if (interleave) {
        LAUNCH_QK_ROPE_KERNEL(256, true);
      } else {
        LAUNCH_QK_ROPE_KERNEL(256, false);
      }
      break;
    default:
      TORCH_CHECK(false, "Unsupported head dimension for fused_qk_rope: ", head_dim);
  }

#undef LAUNCH_QK_ROPE_KERNEL
}

}  // namespace at::native::xpu
