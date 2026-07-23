/* Copyright 2026 SGLang Team. All Rights Reserved.
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * This file adapts the Inkling BMG MoE gate kernels from
 * /data2/syk/cutlass-sycl/examples/16_bmg_moe_gate for the sgl-kernel XPU
 * extension ABI.
 */

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>
#include <vector>

#include "Utils.h"

namespace {

using bf16_t = sycl::ext::oneapi::bfloat16;

constexpr int kRoutedExperts = 256;
constexpr int kSharedExperts = 2;
constexpr int kTotalExperts = kRoutedExperts + kSharedExperts;
constexpr int kTopK = 6;
constexpr int kTopAndShared = kTopK + kSharedExperts;
constexpr int kSubGroupSize = 32;
constexpr int kValuesPerLane = kRoutedExperts / kSubGroupSize;

constexpr int kGateHidden = 6144;
constexpr int kGateLogitsPad = 264;
constexpr int kGateThreads = 256;
constexpr int kGateFusedMaxTokens = 64;

inline float sigmoid_host(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

inline float sigmoid_device(float x) {
  return 1.0f / (1.0f + sycl::native::exp(-x));
}

inline bool score_better(float score, int idx, float best_score, int best_idx) {
  return score > best_score || (score == best_score && idx < best_idx);
}

inline uint16_t f32_to_bf16_rne_device(float x) {
  uint32_t bits = sycl::bit_cast<uint32_t>(x);
  uint32_t lsb = (bits >> 16) & 1u;
  return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

inline int32_t pack_routed_device(int32_t expert, float weight) {
  return static_cast<int32_t>(
      (static_cast<uint32_t>(expert) << 16) | static_cast<uint32_t>(f32_to_bf16_rne_device(weight)));
}

inline float gate_bf16_to_float(bf16_t x) {
  return static_cast<float>(x);
}

struct GateParams {
  float const* logits = nullptr;
  float const* bias = nullptr;
  float const* global_scale = nullptr;
  float* routed_weights = nullptr;
  float* shared_weights = nullptr;
  int32_t* indices = nullptr;
  int32_t* packed = nullptr;
  int64_t tokens = 0;
  int64_t logits_stride = kTotalExperts;
  float route_scale = 1.0f;
};

template <bool Packed>
inline void gate_topk_renorm_row(
    GateParams const& params,
    sycl::sub_group sg,
    int lane,
    int64_t row,
    int64_t row_base) {
  float scores[kValuesPerLane];

#pragma unroll
  for (int j = 0; j < kValuesPerLane; ++j) {
    int expert = lane * kValuesPerLane + j;
    float s = sigmoid_device(params.logits[row_base + expert]);
    scores[j] = s + params.bias[expert];
  }

  int selected_idx[kTopK];
  float selected_sigmoid[kTopK];

#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float best_score = -FLT_MAX;
    int best_idx = INT32_MAX;

#pragma unroll
    for (int j = 0; j < kValuesPerLane; ++j) {
      int expert = lane * kValuesPerLane + j;
      float score = scores[j];
      if (score_better(score, expert, best_score, best_idx)) {
        best_score = score;
        best_idx = expert;
      }
    }

#pragma unroll
    for (int offset = kSubGroupSize / 2; offset > 0; offset >>= 1) {
      float other_score = sycl::permute_group_by_xor(sg, best_score, offset);
      int other_idx = sycl::permute_group_by_xor(sg, best_idx, offset);
      if (score_better(other_score, other_idx, best_score, best_idx)) {
        best_score = other_score;
        best_idx = other_idx;
      }
    }

    if (lane == 0) {
      selected_idx[k] = best_idx;
      float bias = params.bias[best_idx];
      float sigmoid = best_score - bias;
      if (bias > 256.0f || bias < -256.0f) {
        sigmoid = sigmoid_device(params.logits[row_base + best_idx]);
      }
      selected_sigmoid[k] = sigmoid;
    }

    int owner_lane = best_idx / kValuesPerLane;
    int owner_j = best_idx - owner_lane * kValuesPerLane;
    if (lane == owner_lane) {
      scores[owner_j] = -FLT_MAX;
    }
  }

  if (lane == 0) {
    float shared0 = sigmoid_device(params.logits[row_base + kRoutedExperts]);
    float shared1 = sigmoid_device(params.logits[row_base + kRoutedExperts + 1]);
    float sum = shared0 + shared1;

#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      sum += selected_sigmoid[k];
    }

    float scale = params.route_scale * params.global_scale[0] / sum;

#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      float weight = selected_sigmoid[k] * scale;
      if constexpr (Packed) {
        params.packed[row * kTopK + k] = pack_routed_device(selected_idx[k], weight);
      } else {
        params.routed_weights[row * kTopK + k] = weight;
        params.indices[row * kTopK + k] = selected_idx[k];
      }
    }
    params.shared_weights[row * kSharedExperts] = shared0 * scale;
    params.shared_weights[row * kSharedExperts + 1] = shared1 * scale;
  }
}

template <bool Packed, int RowsPerWorkGroup>
class GateTopKRenormKernel {
 public:
  explicit GateTopKRenormKernel(GateParams params) : params_(params) {}

  [[sycl::reqd_sub_group_size(kSubGroupSize)]]
  void operator()(sycl::nd_item<1> item) const {
    sycl::sub_group sg = item.get_sub_group();
    int lane = static_cast<int>(sg.get_local_id());
    int row_in_group = static_cast<int>(sg.get_group_id());
    int64_t row = static_cast<int64_t>(item.get_group(0)) * RowsPerWorkGroup + row_in_group;
    if (row >= params_.tokens) {
      return;
    }

    int64_t row_base = row * params_.logits_stride;
    gate_topk_renorm_row<Packed>(params_, sg, lane, row, row_base);
  }

 private:
  GateParams params_;
};

template <bool Packed, int RowsPerWorkGroup>
sycl::event launch_gate_topk_renorm_static(sycl::queue& queue, GateParams const& params) {
  if (params.tokens == 0) {
    return {};
  }

  int64_t groups = (params.tokens + RowsPerWorkGroup - 1) / RowsPerWorkGroup;
  sycl::range<1> local(static_cast<std::size_t>(RowsPerWorkGroup * kSubGroupSize));
  sycl::range<1> global(static_cast<std::size_t>(groups * RowsPerWorkGroup * kSubGroupSize));
  GateTopKRenormKernel<Packed, RowsPerWorkGroup> kernel(params);

  return queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>(global, local), kernel);
  });
}

template <bool Packed>
sycl::event launch_gate_topk_renorm(sycl::queue& queue, GateParams const& params, int rows_per_workgroup = 0) {
  if (rows_per_workgroup == 0) {
    rows_per_workgroup = params.tokens <= 16384 ? 2 : 1;
  }

  switch (rows_per_workgroup) {
    case 1:
      return launch_gate_topk_renorm_static<Packed, 1>(queue, params);
    case 2:
      return launch_gate_topk_renorm_static<Packed, 2>(queue, params);
    case 4:
      return launch_gate_topk_renorm_static<Packed, 4>(queue, params);
    case 8:
      return launch_gate_topk_renorm_static<Packed, 8>(queue, params);
    default:
      TORCH_CHECK(false, "rows_per_workgroup must be one of {0, 1, 2, 4, 8}");
  }
}

inline sycl::event launch_gate_topk_renorm(
    sycl::queue& queue,
    GateParams const& params,
    bool packed,
    int rows_per_workgroup = 0) {
  return packed ? launch_gate_topk_renorm<true>(queue, params, rows_per_workgroup)
                : launch_gate_topk_renorm<false>(queue, params, rows_per_workgroup);
}

struct GateGemvParams {
  bf16_t const* x = nullptr;
  bf16_t const* weight = nullptr;
  float* logits = nullptr;
  float const* bias = nullptr;
  float const* global_scale = nullptr;
  float* routed_weights = nullptr;
  float* shared_weights = nullptr;
  int32_t* indices = nullptr;
  int32_t* packed = nullptr;
  int32_t* ticket = nullptr;
  int64_t tokens = 0;
  float route_scale = 1.0f;
};

template <int ExpertsPerWorkGroup, int SubGroupSize, bool Fused = false, bool Packed = false>
class GateGemvKernel {
 public:
  static constexpr int kWarps = kGateThreads / SubGroupSize;

  GateGemvParams params;
  sycl::local_accessor<bf16_t, 1> smem_weight;
  sycl::local_accessor<float, 1> smem_partials;
  sycl::local_accessor<int32_t, 1> smem_ticket;

  [[sycl::reqd_sub_group_size(SubGroupSize)]]
  void operator()(sycl::nd_item<1> item) const {
    static_assert(!Fused || SubGroupSize == kSubGroupSize);

    sycl::sub_group sg = item.get_sub_group();
    int tid = static_cast<int>(item.get_local_id(0));
    int lane = static_cast<int>(sg.get_local_id());
    int warp = static_cast<int>(sg.get_group_id());
    int expert0 = static_cast<int>(item.get_group(0)) * ExpertsPerWorkGroup;
    int experts_this_group =
        expert0 + ExpertsPerWorkGroup <= kTotalExperts ? ExpertsPerWorkGroup : kTotalExperts - expert0;

    if (params.tokens > 0 && params.tokens <= 4) {
      float acc[4][ExpertsPerWorkGroup];
#pragma unroll
      for (int token = 0; token < 4; ++token) {
#pragma unroll
        for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
          acc[token][j] = 0.0f;
        }
      }

      for (int k = tid; k < kGateHidden; k += kGateThreads) {
        float x_val[4];
#pragma unroll
        for (int token = 0; token < 4; ++token) {
          x_val[token] = 0.0f;
          if (token < params.tokens) {
            x_val[token] = gate_bf16_to_float(params.x[static_cast<int64_t>(token) * kGateHidden + k]);
          }
        }

#pragma unroll
        for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
          if (j < experts_this_group) {
            bf16_t const* w_row = params.weight + static_cast<int64_t>(expert0 + j) * kGateHidden;
            float wv = gate_bf16_to_float(w_row[k]);
#pragma unroll
            for (int token = 0; token < 4; ++token) {
              if (token < params.tokens) {
                acc[token][j] = sycl::fma(x_val[token], wv, acc[token][j]);
              }
            }
          }
        }
      }

#pragma unroll
      for (int token = 0; token < 4; ++token) {
#pragma unroll
        for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
          float reduced = sycl::reduce_over_group(sg, acc[token][j], sycl::plus<float>());
          if (lane == 0) {
            smem_partials[warp * (4 * ExpertsPerWorkGroup) + token * ExpertsPerWorkGroup + j] = reduced;
          }
        }
      }

      item.barrier(sycl::access::fence_space::local_space);

      if (warp == 0) {
        int total_outputs = static_cast<int>(params.tokens) * ExpertsPerWorkGroup;
        if (lane < total_outputs) {
          int token_out = lane / ExpertsPerWorkGroup;
          int expert_j = lane - token_out * ExpertsPerWorkGroup;
          float sum = 0.0f;
#pragma unroll
          for (int w = 0; w < kWarps; ++w) {
            int offset = w * (4 * ExpertsPerWorkGroup) + token_out * ExpertsPerWorkGroup + expert_j;
            sum += smem_partials[offset];
          }
          if (expert_j < experts_this_group) {
            params.logits[static_cast<int64_t>(token_out) * kGateLogitsPad + expert0 + expert_j] = sum;
          }
        }
      }
    } else {
#pragma unroll
      for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
        if (j < experts_this_group) {
          for (int k = tid; k < kGateHidden; k += kGateThreads) {
            smem_weight[j * kGateHidden + k] = params.weight[static_cast<int64_t>(expert0 + j) * kGateHidden + k];
          }
        }
      }
      item.barrier(sycl::access::fence_space::local_space);

      int warps_per_token = 1;
      if (params.tokens <= 1) {
        warps_per_token = kWarps;
      } else if (params.tokens <= 2) {
        warps_per_token = kWarps / 2;
      } else if (params.tokens <= 4) {
        warps_per_token = kWarps / 4;
      } else if (params.tokens <= 8) {
        warps_per_token = kWarps / 8;
      }
      warps_per_token = warps_per_token < 1 ? 1 : warps_per_token;

      if (warps_per_token > 1) {
        int token = warp / warps_per_token;
        int slice = warp - token * warps_per_token;
        int span = kGateHidden / warps_per_token;
        int k_begin = slice * span;
        int k_end = k_begin + span;

        if (token < params.tokens) {
          bf16_t const* x_row = params.x + static_cast<int64_t>(token) * kGateHidden;
#pragma unroll
          for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
            float acc = 0.0f;
            if (j < experts_this_group) {
              for (int k = k_begin + lane; k < k_end; k += SubGroupSize) {
                float xv = gate_bf16_to_float(x_row[k]);
                float wv = gate_bf16_to_float(smem_weight[j * kGateHidden + k]);
                acc = sycl::fma(xv, wv, acc);
              }
            }
            float reduced = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
            if (lane == 0) {
              smem_partials[warp * ExpertsPerWorkGroup + j] = reduced;
            }
          }
        }

        item.barrier(sycl::access::fence_space::local_space);

        if (warp == 0) {
          int total_outputs = static_cast<int>(params.tokens) * ExpertsPerWorkGroup;
          if (lane < total_outputs) {
            int token_out = lane / ExpertsPerWorkGroup;
            int expert_j = lane - token_out * ExpertsPerWorkGroup;
            float sum = 0.0f;
            for (int s = 0; s < warps_per_token; ++s) {
              sum += smem_partials[(token_out * warps_per_token + s) * ExpertsPerWorkGroup + expert_j];
            }
            if (expert_j < experts_this_group) {
              params.logits[static_cast<int64_t>(token_out) * kGateLogitsPad + expert0 + expert_j] = sum;
            }
          }
        }
      } else {
        for (int64_t token = warp; token < params.tokens; token += kWarps) {
          bf16_t const* x_row = params.x + token * kGateHidden;
          float acc[ExpertsPerWorkGroup];
#pragma unroll
          for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
            acc[j] = 0.0f;
          }

          for (int k = lane; k < kGateHidden; k += SubGroupSize) {
            float xv = gate_bf16_to_float(x_row[k]);
#pragma unroll
            for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
              if (j < experts_this_group) {
                float wv = gate_bf16_to_float(smem_weight[j * kGateHidden + k]);
                acc[j] = sycl::fma(xv, wv, acc[j]);
              }
            }
          }

#pragma unroll
          for (int j = 0; j < ExpertsPerWorkGroup; ++j) {
            float reduced = sycl::reduce_over_group(sg, acc[j], sycl::plus<float>());
            if (lane == 0 && j < experts_this_group) {
              params.logits[token * kGateLogitsPad + expert0 + j] = reduced;
            }
          }
        }
      }
    }

    if constexpr (Fused) {
      item.barrier(sycl::access::fence_space::global_and_local);
      sycl::atomic_fence(sycl::memory_order::release, sycl::memory_scope::device);

      if (tid == 0) {
        sycl::atomic_ref<int32_t,
                         sycl::memory_order::acq_rel,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            counter(params.ticket[0]);
        smem_ticket[0] = counter.fetch_add(1);
      }

      item.barrier(sycl::access::fence_space::local_space);
      int32_t ticket_value = smem_ticket[0];
      int32_t last_ticket = static_cast<int32_t>(item.get_group_range(0)) - 1;
      if (ticket_value != last_ticket) {
        return;
      }

      sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);

      GateParams gate_params;
      gate_params.logits = params.logits;
      gate_params.bias = params.bias;
      gate_params.global_scale = params.global_scale;
      gate_params.routed_weights = params.routed_weights;
      gate_params.shared_weights = params.shared_weights;
      gate_params.indices = params.indices;
      gate_params.packed = params.packed;
      gate_params.tokens = params.tokens;
      gate_params.logits_stride = kGateLogitsPad;
      gate_params.route_scale = params.route_scale;

      for (int64_t row = warp; row < params.tokens; row += kWarps) {
        gate_topk_renorm_row<Packed>(gate_params, sg, lane, row, row * static_cast<int64_t>(kGateLogitsPad));
      }

      item.barrier(sycl::access::fence_space::global_and_local);
      if (tid == 0) {
        sycl::atomic_ref<int32_t,
                         sycl::memory_order::acq_rel,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            counter(params.ticket[0]);
        counter.store(0, sycl::memory_order::release, sycl::memory_scope::device);
      }
    }
  }
};

template <int ExpertsPerWorkGroup, int SubGroupSize, bool Fused = false, bool Packed = false>
sycl::event launch_gate_gemv_static(sycl::queue& queue, GateGemvParams const& params) {
  if (params.tokens == 0) {
    return {};
  }

  static_assert(ExpertsPerWorkGroup == 1 || ExpertsPerWorkGroup == 2 || ExpertsPerWorkGroup == 4);
  static_assert(SubGroupSize == 16 || SubGroupSize == 32);
  static_assert(!Fused || SubGroupSize == kSubGroupSize);
  if constexpr (Fused) {
    TORCH_CHECK(params.tokens <= kGateFusedMaxTokens, "fused gate GEMV supports at most 64 tokens");
    TORCH_CHECK(params.ticket != nullptr, "fused gate GEMV requires a non-null ticket pointer");
  }
  int64_t groups = (kTotalExperts + ExpertsPerWorkGroup - 1) / ExpertsPerWorkGroup;
  sycl::range<1> local(static_cast<std::size_t>(kGateThreads));
  sycl::range<1> global(static_cast<std::size_t>(groups * kGateThreads));

  return queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<bf16_t, 1> smem_weight(
        sycl::range<1>(static_cast<std::size_t>(ExpertsPerWorkGroup * kGateHidden)), cgh);
    sycl::local_accessor<float, 1> smem_partials(
        sycl::range<1>(static_cast<std::size_t>((kGateThreads / SubGroupSize) * 4 * ExpertsPerWorkGroup)), cgh);
    sycl::local_accessor<int32_t, 1> smem_ticket(sycl::range<1>(1), cgh);
    GateGemvKernel<ExpertsPerWorkGroup, SubGroupSize, Fused, Packed> kernel{
        params, smem_weight, smem_partials, smem_ticket};
    cgh.parallel_for<GateGemvKernel<ExpertsPerWorkGroup, SubGroupSize, Fused, Packed>>(
        sycl::nd_range<1>(global, local), kernel);
  });
}

inline int default_gate_gemv_experts_per_workgroup(int requested) {
  int value = requested == 0 ? 1 : requested;
  TORCH_CHECK(value == 1 || value == 2 || value == 4, "experts_per_workgroup must be one of {0, 1, 2, 4}");
  return value;
}

inline int default_gate_gemv_subgroup_size(int requested) {
  int value = requested == 0 ? 32 : requested;
  TORCH_CHECK(value == 16 || value == 32, "subgroup_size must be one of {0, 16, 32}");
  return value;
}

template <int ExpertsPerWorkGroup>
sycl::event launch_gate_gemv_experts(sycl::queue& queue, GateGemvParams const& params, int subgroup_size) {
  subgroup_size = default_gate_gemv_subgroup_size(subgroup_size);
  switch (subgroup_size) {
    case 16:
      return launch_gate_gemv_static<ExpertsPerWorkGroup, 16, false, false>(queue, params);
    case 32:
      return launch_gate_gemv_static<ExpertsPerWorkGroup, 32, false, false>(queue, params);
    default:
      TORCH_CHECK(false, "subgroup_size must be one of {0, 16, 32}");
  }
}

inline sycl::event launch_gate_gemv(
    sycl::queue& queue,
    GateGemvParams const& params,
    int experts_per_workgroup = 0,
    int subgroup_size = 0) {
  experts_per_workgroup = default_gate_gemv_experts_per_workgroup(experts_per_workgroup);
  switch (experts_per_workgroup) {
    case 1:
      return launch_gate_gemv_experts<1>(queue, params, subgroup_size);
    case 2:
      return launch_gate_gemv_experts<2>(queue, params, subgroup_size);
    case 4:
      return launch_gate_gemv_experts<4>(queue, params, subgroup_size);
    default:
      TORCH_CHECK(false, "experts_per_workgroup must be one of {0, 1, 2, 4}");
  }
}

template <bool Packed, int ExpertsPerWorkGroup>
sycl::event launch_gate_gemv_fused_experts(
    sycl::queue& queue,
    GateGemvParams const& params,
    int subgroup_size) {
  subgroup_size = default_gate_gemv_subgroup_size(subgroup_size);
  TORCH_CHECK(subgroup_size == kSubGroupSize, "fused gate GEMV requires subgroup_size 32");
  return launch_gate_gemv_static<ExpertsPerWorkGroup, kSubGroupSize, true, Packed>(queue, params);
}

template <bool Packed>
sycl::event launch_gate_gemv_fused(
    sycl::queue& queue,
    GateGemvParams const& params,
    int experts_per_workgroup = 0,
    int subgroup_size = 0) {
  experts_per_workgroup = default_gate_gemv_experts_per_workgroup(experts_per_workgroup);
  switch (experts_per_workgroup) {
    case 1:
      return launch_gate_gemv_fused_experts<Packed, 1>(queue, params, subgroup_size);
    case 2:
      return launch_gate_gemv_fused_experts<Packed, 2>(queue, params, subgroup_size);
    case 4:
      return launch_gate_gemv_fused_experts<Packed, 4>(queue, params, subgroup_size);
    default:
      TORCH_CHECK(false, "experts_per_workgroup must be one of {0, 1, 2, 4}");
  }
}

inline sycl::event launch_gate_gemv_fused(
    sycl::queue& queue,
    GateGemvParams const& params,
    bool packed,
    int experts_per_workgroup = 0,
    int subgroup_size = 0) {
  return packed ? launch_gate_gemv_fused<true>(queue, params, experts_per_workgroup, subgroup_size)
                : launch_gate_gemv_fused<false>(queue, params, experts_per_workgroup, subgroup_size);
}

void check_gate_logits(const at::Tensor& logits) {
  CHECK_DEVICE(logits);
  TORCH_CHECK(logits.scalar_type() == at::ScalarType::Float, "logits must be float32");
  TORCH_CHECK(logits.dim() == 2, "logits must have shape [tokens, experts]");
  TORCH_CHECK(logits.size(1) >= kTotalExperts, "logits second dimension must be at least 258");
  TORCH_CHECK(logits.stride(1) == 1, "logits last dimension must be contiguous");
}

void check_bias_scale(const at::Tensor& bias, const at::Tensor& global_scale, const c10::Device& device) {
  CHECK_DEVICE(bias);
  CHECK_DEVICE(global_scale);
  TORCH_CHECK(bias.device() == device, "bias must be on the same device as inputs");
  TORCH_CHECK(global_scale.device() == device, "global_scale must be on the same device as inputs");
  TORCH_CHECK(bias.scalar_type() == at::ScalarType::Float, "bias must be float32");
  TORCH_CHECK(global_scale.scalar_type() == at::ScalarType::Float, "global_scale must be float32");
  TORCH_CHECK(bias.sizes() == at::IntArrayRef({kRoutedExperts}), "bias must have shape [256]");
  TORCH_CHECK(global_scale.numel() == 1, "global_scale must contain one value");
  TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
  TORCH_CHECK(global_scale.is_contiguous(), "global_scale must be contiguous");
}

void check_gemv_inputs(const at::Tensor& x, const at::Tensor& weight) {
  CHECK_DEVICE(x);
  CHECK_DEVICE(weight);
  TORCH_CHECK(x.device() == weight.device(), "x and weight must be on the same device");
  TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, "x must be bfloat16");
  TORCH_CHECK(weight.scalar_type() == at::ScalarType::BFloat16, "weight must be bfloat16");
  TORCH_CHECK(x.dim() == 2, "x must have shape [tokens, 6144]");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [>=258, 6144]");
  TORCH_CHECK(x.size(1) == kGateHidden, "x second dimension must be 6144");
  TORCH_CHECK(weight.size(0) >= kTotalExperts, "weight first dimension must be at least 258");
  TORCH_CHECK(weight.size(1) == kGateHidden, "weight second dimension must be 6144");
  TORCH_CHECK(x.stride(1) == 1 && x.stride(0) == kGateHidden, "x must be contiguous");
  TORCH_CHECK(weight.stride(1) == 1 && weight.stride(0) == kGateHidden, "weight must be contiguous");
}

GateGemvParams make_gemv_params(
    const at::Tensor& x,
    const at::Tensor& weight,
    at::Tensor& logits,
    const at::Tensor* bias,
    const at::Tensor* global_scale,
    at::Tensor* routed_weights,
    at::Tensor* shared_weights,
    at::Tensor* indices,
    at::Tensor* packed,
    at::Tensor* ticket,
    double route_scale) {
  GateGemvParams params;
  params.x = reinterpret_cast<bf16_t const*>(x.data_ptr<at::BFloat16>());
  params.weight = reinterpret_cast<bf16_t const*>(weight.data_ptr<at::BFloat16>());
  params.logits = logits.data_ptr<float>();
  params.bias = bias == nullptr ? nullptr : bias->data_ptr<float>();
  params.global_scale = global_scale == nullptr ? nullptr : global_scale->data_ptr<float>();
  params.routed_weights = routed_weights == nullptr ? nullptr : routed_weights->data_ptr<float>();
  params.shared_weights = shared_weights == nullptr ? nullptr : shared_weights->data_ptr<float>();
  params.indices = indices == nullptr ? nullptr : indices->data_ptr<int32_t>();
  params.packed = packed == nullptr ? nullptr : packed->data_ptr<int32_t>();
  params.ticket = ticket == nullptr ? nullptr : ticket->data_ptr<int32_t>();
  params.tokens = x.size(0);
  params.route_scale = static_cast<float>(route_scale);
  return params;
}

}  // namespace

std::vector<at::Tensor> inkling_moe_gate_topk_renorm(
    at::Tensor& logits,
    at::Tensor& bias,
    at::Tensor& global_scale,
    double route_scale,
    bool return_packed,
    int64_t rows_per_workgroup) {
  check_gate_logits(logits);
  check_bias_scale(bias, global_scale, logits.device());
  TORCH_CHECK(
      rows_per_workgroup == 0 || rows_per_workgroup == 1 || rows_per_workgroup == 2 || rows_per_workgroup == 4 ||
          rows_per_workgroup == 8,
      "rows_per_workgroup must be one of {0, 1, 2, 4, 8}");

  int64_t tokens = logits.size(0);
  auto float_options = logits.options().dtype(at::ScalarType::Float);
  auto int_options = logits.options().dtype(at::ScalarType::Int);
  at::Tensor shared_weights = at::empty({tokens, kSharedExperts}, float_options);

  GateParams params;
  params.logits = logits.data_ptr<float>();
  params.bias = bias.data_ptr<float>();
  params.global_scale = global_scale.data_ptr<float>();
  params.shared_weights = shared_weights.data_ptr<float>();
  params.tokens = tokens;
  params.logits_stride = logits.stride(0);
  params.route_scale = static_cast<float>(route_scale);

  auto queue = at::xpu::getCurrentXPUStream().queue();
  if (return_packed) {
    at::Tensor packed = at::empty({tokens, kTopK}, int_options);
    params.packed = packed.data_ptr<int32_t>();
    if (tokens > 0) {
      launch_gate_topk_renorm(queue, params, true, static_cast<int>(rows_per_workgroup));
    }
    return {packed, shared_weights};
  }

  at::Tensor routed_weights = at::empty({tokens, kTopK}, float_options);
  at::Tensor indices = at::empty({tokens, kTopK}, int_options);
  params.routed_weights = routed_weights.data_ptr<float>();
  params.indices = indices.data_ptr<int32_t>();
  if (tokens > 0) {
    launch_gate_topk_renorm(queue, params, false, static_cast<int>(rows_per_workgroup));
  }
  return {routed_weights, indices, shared_weights};
}

at::Tensor inkling_moe_gate_gemv(
    at::Tensor& x,
    at::Tensor& weight,
    int64_t experts_per_workgroup,
    int64_t subgroup_size) {
  check_gemv_inputs(x, weight);
  TORCH_CHECK(
      experts_per_workgroup == 0 || experts_per_workgroup == 1 || experts_per_workgroup == 2 ||
          experts_per_workgroup == 4,
      "experts_per_workgroup must be one of {0, 1, 2, 4}");
  TORCH_CHECK(
      subgroup_size == 0 || subgroup_size == 16 || subgroup_size == 32,
      "subgroup_size must be one of {0, 16, 32}");

  at::Tensor logits = at::empty({x.size(0), kGateLogitsPad}, x.options().dtype(at::ScalarType::Float));
  if (x.size(0) > 0) {
    auto params = make_gemv_params(
        x,
        weight,
        logits,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        1.0);
    auto queue = at::xpu::getCurrentXPUStream().queue();
    launch_gate_gemv(queue, params, static_cast<int>(experts_per_workgroup), static_cast<int>(subgroup_size));
  }
  return logits.slice(/*dim=*/1, /*start=*/0, /*end=*/kTotalExperts);
}

std::vector<at::Tensor> inkling_moe_gate_gemv_fused(
    at::Tensor& x,
    at::Tensor& weight,
    at::Tensor& bias,
    at::Tensor& global_scale,
    at::Tensor& workspace,
    at::Tensor& ticket,
    double route_scale,
    bool return_packed,
    int64_t experts_per_workgroup,
    int64_t subgroup_size) {
  check_gemv_inputs(x, weight);
  check_bias_scale(bias, global_scale, x.device());
  CHECK_DEVICE(workspace);
  CHECK_DEVICE(ticket);
  TORCH_CHECK(workspace.device() == x.device(), "workspace must be on the same device as x");
  TORCH_CHECK(ticket.device() == x.device(), "ticket must be on the same device as x");
  TORCH_CHECK(workspace.scalar_type() == at::ScalarType::Float, "workspace must be float32");
  TORCH_CHECK(ticket.scalar_type() == at::ScalarType::Int, "ticket must be int32");
  TORCH_CHECK(
      workspace.sizes() == at::IntArrayRef({kGateFusedMaxTokens, kGateLogitsPad}),
      "workspace must have shape [64, 264]");
  TORCH_CHECK(ticket.sizes() == at::IntArrayRef({1}), "ticket must have shape [1]");
  TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");
  TORCH_CHECK(ticket.is_contiguous(), "ticket must be contiguous");
  TORCH_CHECK(x.size(0) <= kGateFusedMaxTokens, "fused gate GEMV supports at most 64 tokens");
  TORCH_CHECK(
      experts_per_workgroup == 0 || experts_per_workgroup == 1 || experts_per_workgroup == 2 ||
          experts_per_workgroup == 4,
      "experts_per_workgroup must be one of {0, 1, 2, 4}");
  TORCH_CHECK(subgroup_size == 0 || subgroup_size == 32, "fused gate GEMV requires subgroup_size 32");

  int64_t tokens = x.size(0);
  auto float_options = x.options().dtype(at::ScalarType::Float);
  auto int_options = x.options().dtype(at::ScalarType::Int);
  at::Tensor shared_weights = at::empty({tokens, kSharedExperts}, float_options);

  if (return_packed) {
    at::Tensor packed = at::empty({tokens, kTopK}, int_options);
    if (tokens > 0) {
      auto params = make_gemv_params(
          x, weight, workspace, &bias, &global_scale, nullptr, &shared_weights, nullptr, &packed, &ticket, route_scale);
      auto queue = at::xpu::getCurrentXPUStream().queue();
      launch_gate_gemv_fused(
          queue,
          params,
          true,
          static_cast<int>(experts_per_workgroup),
          static_cast<int>(subgroup_size));
    }
    return {packed, shared_weights};
  }

  at::Tensor routed_weights = at::empty({tokens, kTopK}, float_options);
  at::Tensor indices = at::empty({tokens, kTopK}, int_options);
  if (tokens > 0) {
    auto params = make_gemv_params(
        x,
        weight,
        workspace,
        &bias,
        &global_scale,
        &routed_weights,
        &shared_weights,
        &indices,
        nullptr,
        &ticket,
        route_scale);
    auto queue = at::xpu::getCurrentXPUStream().queue();
    launch_gate_gemv_fused(
        queue,
        params,
        false,
        static_cast<int>(experts_per_workgroup),
        static_cast<int>(subgroup_size));
  }
  return {routed_weights, indices, shared_weights};
}
