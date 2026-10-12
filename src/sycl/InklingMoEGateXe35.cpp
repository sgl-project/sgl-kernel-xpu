/* Copyright 2026 SGLang Team. All Rights Reserved.
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

#define SYCL_INTEL_TARGET 35

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <vector>

#include "SGLKernelPerf.h"
#include "Utils.h"
#include "sgl_kernel_export.h"
#include "sycl/kernels/inkling_moe_gate/inkling_moe_gate_projection.hpp"

namespace {

using bf16_t = sycl::ext::oneapi::bfloat16;
using namespace inkling_moe_gate_impl;

constexpr int kRoutedExperts = 256;
constexpr int kSharedExperts = 2;
constexpr int kTopK = 6;
constexpr int kSubgroupSize = 32;
constexpr int kValuesPerLane = kRoutedExperts / kSubgroupSize;

inline float sigmoid_device(float value) {
  return sycl::native::recip(1.0f + sycl::native::exp(-value));
}

inline bool score_better(float score, int index, float best_score, int best_index) {
  return score > best_score || (score == best_score && index < best_index);
}

inline uint16_t f32_to_bf16_rne(float value) {
  uint32_t bits = sycl::bit_cast<uint32_t>(value);
  uint32_t lsb = (bits >> 16) & 1u;
  return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

inline int32_t pack_route(int32_t expert, float weight) {
  return static_cast<int32_t>((static_cast<uint32_t>(expert) << 16) | static_cast<uint32_t>(f32_to_bf16_rne(weight)));
}

template <typename T>
inline float load_float(const T* pointer, int64_t index) {
  return static_cast<float>(pointer[index]);
}

struct TopKParams {
  const void* logits;
  const float* bias;
  const float* global_scale;
  float* routed_weights;
  int32_t* indices;
  float* shared_weights;
  int32_t* packed;
  int64_t tokens;
  int64_t stride;
  float route_scale;
};

template <typename T, bool Packed>
class InklingGateTopKKernel {
 public:
  explicit InklingGateTopKKernel(TopKParams params) : params_(params) {}

  [[sycl::reqd_sub_group_size(kSubgroupSize)]]
  void operator()(sycl::nd_item<1> item) const {
    auto subgroup = item.get_sub_group();
    const int lane = static_cast<int>(subgroup.get_local_id());
    const int subgroup_id = static_cast<int>(subgroup.get_group_id());
    const int64_t row = static_cast<int64_t>(item.get_group(0)) * 2 + subgroup_id;
    if (row >= params_.tokens) {
      return;
    }

    const T* logits = static_cast<const T*>(params_.logits);
    const int64_t base = row * params_.stride;
    float ranking[kValuesPerLane];
    int ranking_indices[kValuesPerLane];

#pragma unroll
    for (int item_index = 0; item_index < kValuesPerLane; ++item_index) {
      const int expert = item_index * kSubgroupSize + lane;
      ranking[item_index] = sigmoid_device(load_float(logits, base + expert)) + params_.bias[expert];
      ranking_indices[item_index] = expert;
    }

    // Sort each lane's eight candidates once. The six subgroup selections can
    // then merge only the current lane heads instead of rescanning all eight.
#pragma unroll
    for (int item_index = 1; item_index < kValuesPerLane; ++item_index) {
#pragma unroll
      for (int position = item_index; position > 0; --position) {
        if (score_better(
                ranking[position], ranking_indices[position], ranking[position - 1], ranking_indices[position - 1])) {
          const float score = ranking[position - 1];
          const int index = ranking_indices[position - 1];
          ranking[position - 1] = ranking[position];
          ranking_indices[position - 1] = ranking_indices[position];
          ranking[position] = score;
          ranking_indices[position] = index;
        }
      }
    }

    int selected_indices[kTopK];
    float selected_weights[kTopK];
#pragma unroll
    for (int selected = 0; selected < kTopK; ++selected) {
      float best_score = ranking[0];
      int best_index = ranking_indices[0];
#pragma unroll
      for (int offset = kSubgroupSize / 2; offset > 0; offset >>= 1) {
        const float other_score = sycl::permute_group_by_xor(subgroup, best_score, offset);
        const int other_index = sycl::permute_group_by_xor(subgroup, best_index, offset);
        if (score_better(other_score, other_index, best_score, best_index)) {
          best_score = other_score;
          best_index = other_index;
        }
      }
      if (lane == 0) {
        selected_indices[selected] = best_index;
        const float correction = params_.bias[best_index];
        selected_weights[selected] = sycl::fabs(correction) <= 256.0f
                                         ? best_score - correction
                                         : sigmoid_device(load_float(logits, base + best_index));
      }
      const int owner_lane = best_index % kSubgroupSize;
      if (lane == owner_lane) {
#pragma unroll
        for (int item_index = 0; item_index < kValuesPerLane - 1; ++item_index) {
          ranking[item_index] = ranking[item_index + 1];
          ranking_indices[item_index] = ranking_indices[item_index + 1];
        }
        ranking[kValuesPerLane - 1] = -FLT_MAX;
        ranking_indices[kValuesPerLane - 1] = INT_MAX;
      }
    }

    if (lane != 0) {
      return;
    }
    const float shared0 = sigmoid_device(load_float(logits, base + kRoutedExperts));
    const float shared1 = sigmoid_device(load_float(logits, base + kRoutedExperts + 1));
    float denominator = shared0 + shared1;
#pragma unroll
    for (int selected = 0; selected < kTopK; ++selected) {
      denominator += selected_weights[selected];
    }
    const float scale = params_.route_scale * params_.global_scale[0] / denominator;
#pragma unroll
    for (int selected = 0; selected < kTopK; ++selected) {
      const float weight = selected_weights[selected] * scale;
      const int64_t output_index = row * kTopK + selected;
      if constexpr (Packed) {
        params_.packed[output_index] = pack_route(selected_indices[selected], weight);
      } else {
        params_.routed_weights[output_index] = weight;
        params_.indices[output_index] = selected_indices[selected];
      }
    }
    params_.shared_weights[row * kSharedExperts] = shared0 * scale;
    params_.shared_weights[row * kSharedExperts + 1] = shared1 * scale;
  }

 private:
  TopKParams params_;
};

template <typename T, bool Packed>
void launch_topk(sycl::queue& queue, const TopKParams& params) {
  constexpr int kRowsPerWorkgroup = 2;
  const int64_t groups = (params.tokens + kRowsPerWorkgroup - 1) / kRowsPerWorkgroup;
  const sycl::range<1> local(kRowsPerWorkgroup * kSubgroupSize);
  const sycl::range<1> global(groups * kRowsPerWorkgroup * kSubgroupSize);
  queue.submit([&](sycl::handler& handler) {
    handler.parallel_for(sycl::nd_range<1>(global, local), InklingGateTopKKernel<T, Packed>(params));
  });
}

template <bool Packed>
void dispatch_topk(sycl::queue& queue, const TopKParams& params, at::ScalarType dtype) {
  switch (dtype) {
    case at::ScalarType::Float:
      return launch_topk<float, Packed>(queue, params);
    case at::ScalarType::BFloat16:
      return launch_topk<bf16_t, Packed>(queue, params);
    case at::ScalarType::Half:
      return launch_topk<sycl::half, Packed>(queue, params);
    default:
      TORCH_CHECK(false, "Inkling gate top-k supports float32, bfloat16, and float16 logits");
  }
}

void check_projection_inputs(const at::Tensor& input, const at::Tensor& weight) {
  CHECK_INPUT(input);
  CHECK_INPUT(weight);
  TORCH_CHECK(input.dim() == 2 && input.size(1) == kHidden, "input must have shape [tokens, 6144]");
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({kPaddedExperts, kHidden}), "weight must have shape [264, 6144]");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::BFloat16 || input.scalar_type() == at::ScalarType::Half,
      "input must be bfloat16 or float16");
  TORCH_CHECK(weight.scalar_type() == input.scalar_type(), "weight dtype must match input");
}

void check_topk_inputs(const at::Tensor& logits, const at::Tensor& bias, const at::Tensor& global_scale) {
  CHECK_DEVICE(logits);
  CHECK_DEVICE(bias);
  CHECK_DEVICE(global_scale);
  TORCH_CHECK(logits.dim() == 2 && logits.size(1) == kActiveExperts, "logits must have shape [tokens, 258]");
  TORCH_CHECK(logits.stride(1) == 1, "logits last dimension must be contiguous");
  TORCH_CHECK(
      logits.scalar_type() == at::ScalarType::Float || logits.scalar_type() == at::ScalarType::BFloat16 ||
          logits.scalar_type() == at::ScalarType::Half,
      "logits must be float32, bfloat16, or float16");
  TORCH_CHECK(bias.sizes() == at::IntArrayRef({kRoutedExperts}), "bias must have shape [256]");
  TORCH_CHECK(bias.scalar_type() == at::ScalarType::Float && bias.is_contiguous(), "bias must be contiguous float32");
  TORCH_CHECK(
      global_scale.scalar_type() == at::ScalarType::Float && global_scale.numel() == 1 && global_scale.is_contiguous(),
      "global_scale must be a contiguous float32 scalar tensor");
  TORCH_CHECK(
      logits.device() == bias.device() && logits.device() == global_scale.device(),
      "all inputs must be on the same device");
}

void dispatch_projection(const at::Tensor& input, const at::Tensor& weight, at::Tensor& output, sycl::queue& queue) {
  if (input.scalar_type() == at::ScalarType::BFloat16) {
    return launch_projection_bf16(input, weight, output, queue);
  }
  return launch_projection_fp16(input, weight, output, queue);
}

}  // namespace

SGL_KERNEL_EXPORT at::Tensor inkling_moe_gate_projection(const at::Tensor& input, const at::Tensor& weight) {
  check_projection_inputs(input, weight);
  auto output = at::empty({input.size(0), kPaddedExperts}, input.options().dtype(at::ScalarType::Float));
  if (input.size(0) == 0) {
    return output.slice(1, 0, kActiveExperts);
  }
  auto queue = at::xpu::getCurrentXPUStream().queue();
#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  const double flops = 2.0 * input.size(0) * kPaddedExperts * kHidden;
  const double bytes = input.numel() * input.element_size() + weight.numel() * weight.element_size() +
                       output.numel() * output.element_size();
  SGL_KERNEL_PERF_SCOPE("inkling_moe_gate_projection", queue, bytes, flops);
#endif
  dispatch_projection(input, weight, output, queue);
  return output.slice(1, 0, kActiveExperts);
}

SGL_KERNEL_EXPORT std::vector<at::Tensor> inkling_moe_gate_topk_renorm(
    const at::Tensor& logits,
    const at::Tensor& bias,
    const at::Tensor& global_scale,
    double route_scale,
    bool return_packed) {
  check_topk_inputs(logits, bias, global_scale);
  const auto float_options = logits.options().dtype(at::ScalarType::Float);
  const auto int_options = logits.options().dtype(at::ScalarType::Int);
  at::Tensor shared_weights = at::empty({logits.size(0), kSharedExperts}, float_options);

  TopKParams params{
      logits.data_ptr(),
      bias.data_ptr<float>(),
      global_scale.data_ptr<float>(),
      nullptr,
      nullptr,
      shared_weights.data_ptr<float>(),
      nullptr,
      logits.size(0),
      logits.stride(0),
      static_cast<float>(route_scale)};
  if (logits.size(0) == 0) {
    if (return_packed) {
      return {at::empty({0, kTopK}, int_options), shared_weights};
    }
    return {at::empty({0, kTopK}, float_options), at::empty({0, kTopK}, int_options), shared_weights};
  }

  auto queue = at::xpu::getCurrentXPUStream().queue();
  if (return_packed) {
    at::Tensor packed = at::empty({logits.size(0), kTopK}, int_options);
    params.packed = packed.data_ptr<int32_t>();
    dispatch_topk<true>(queue, params, logits.scalar_type());
    return {packed, shared_weights};
  }
  at::Tensor routed_weights = at::empty({logits.size(0), kTopK}, float_options);
  at::Tensor indices = at::empty({logits.size(0), kTopK}, int_options);
  params.routed_weights = routed_weights.data_ptr<float>();
  params.indices = indices.data_ptr<int32_t>();
  dispatch_topk<false>(queue, params, logits.scalar_type());
  return {routed_weights, indices, shared_weights};
}

#undef SYCL_INTEL_TARGET
