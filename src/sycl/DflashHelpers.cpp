/* Copyright 2025 SGLang Team. All Rights Reserved.
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * This file adapts the Inkling BMG DFLASH helper kernels from
 * /data2/syk/cutlass-sycl/examples/20_bmg_dflash for the sgl-kernel XPU
 * extension ABI.
 */

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <sycl/sycl.hpp>
#include <tuple>

#include "Utils.h"

namespace {

constexpr int kThreads = 256;
constexpr int kCopyPackBytes = 16;

inline int64_t ceil_div_i64(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

inline int checked_int64_to_int(int64_t value, char const* name, bool allow_zero = false) {
  TORCH_CHECK(
      (allow_zero ? value >= 0 : value > 0) && value <= std::numeric_limits<int>::max(),
      name,
      allow_zero ? " must be non-negative and fit in int32" : " must be positive and fit in int32");
  return static_cast<int>(value);
}

struct MaskedGatherParams {
  int64_t const* req_to_token = nullptr;
  int64_t const* req_pool_indices = nullptr;
  int64_t const* pos2d = nullptr;
  uint8_t const* mask = nullptr;
  int32_t const* out_offsets = nullptr;
  int64_t* out = nullptr;
  int batch = 0;
  int draft_tokens = 0;
  int table_width = 0;
};

struct MaskedGatherKernel {
  MaskedGatherParams p;

  void operator()(sycl::id<1> id) const {
    const int lane = static_cast<int>(id[0]);
    const int total = p.batch * p.draft_tokens;
    if (lane >= total || p.mask[lane] == 0) {
      return;
    }
    const int b = lane / p.draft_tokens;
    const int64_t req_slot = p.req_pool_indices[b];
    const int64_t pos = p.pos2d[lane];
    const int32_t out_idx = p.out_offsets[lane];
    p.out[out_idx] = p.req_to_token[req_slot * p.table_width + pos];
  }
};

void launch_masked_gather(sycl::queue& queue, MaskedGatherParams const& params) {
  const int total = params.batch * params.draft_tokens;
  if (total <= 0) {
    return;
  }
  queue.parallel_for<MaskedGatherKernel>(
      sycl::range<1>(static_cast<std::size_t>(total)), MaskedGatherKernel{params});
}

struct LogitsArgmaxParams {
  float const* logits = nullptr;
  int64_t* out_tokens = nullptr;
  int tokens = 0;
  int vocab = 0;
};

struct LogitsArgmaxKernel {
  LogitsArgmaxParams p;
  sycl::local_accessor<float, 1> values;
  sycl::local_accessor<int32_t, 1> indices;

  void operator()(sycl::nd_item<1> item) const {
    const int token = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    const int local = static_cast<int>(item.get_local_range(0));
    float best = -3.4028234663852886e38f;
    int32_t best_idx = 0;

    float const* row = p.logits + static_cast<int64_t>(token) * p.vocab;
    for (int v = tid; v < p.vocab; v += local) {
      const float value = row[v];
      if (value > best || (value == best && v < best_idx)) {
        best = value;
        best_idx = v;
      }
    }
    values[tid] = best;
    indices[tid] = best_idx;
    item.barrier(sycl::access::fence_space::local_space);

    for (int stride = local / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        const float other = values[tid + stride];
        const int32_t other_idx = indices[tid + stride];
        if (other > values[tid] || (other == values[tid] && other_idx < indices[tid])) {
          values[tid] = other;
          indices[tid] = other_idx;
        }
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    if (tid == 0) {
      p.out_tokens[token] = static_cast<int64_t>(indices[0]);
    }
  }
};

void launch_logits_argmax(sycl::queue& queue, LogitsArgmaxParams const& params) {
  if (params.tokens <= 0 || params.vocab <= 0) {
    return;
  }
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> values(kThreads, cgh);
    sycl::local_accessor<int32_t, 1> indices(kThreads, cgh);
    sycl::range<1> local(kThreads);
    sycl::range<1> global(static_cast<std::size_t>(params.tokens) * kThreads);
    cgh.parallel_for(
        sycl::nd_range<1>(global, local),
        LogitsArgmaxKernel{params, values, indices});
  });
}

constexpr int kNameStrideMinimum = 10;

struct DeviceGuardParams {
  char const* names = nullptr;
  uint8_t* legacy_cuda_guard = nullptr;
  uint8_t* dflash_supported_guard = nullptr;
  int count = 0;
  int stride = 0;
};

struct DeviceGuardKernel {
  DeviceGuardParams p;

  void operator()(sycl::id<1> id) const {
    const int row = static_cast<int>(id[0]);
    if (row >= p.count) {
      return;
    }
    char const* s = p.names + row * p.stride;
    const bool is_cuda = s[0] == 'c' && s[1] == 'u' && s[2] == 'd' && s[3] == 'a';
    const bool is_xpu = s[0] == 'x' && s[1] == 'p' && s[2] == 'u';
    const bool is_level_zero = s[0] == 'l' && s[1] == 'e' && s[2] == 'v' && s[3] == 'e' &&
        s[4] == 'l' && s[5] == '_' && s[6] == 'z' && s[7] == 'e' && s[8] == 'r' && s[9] == 'o';

    p.legacy_cuda_guard[row] = static_cast<uint8_t>(is_cuda);
    p.dflash_supported_guard[row] = static_cast<uint8_t>(is_cuda || is_xpu || is_level_zero);
  }
};

void launch_device_guard(sycl::queue& queue, DeviceGuardParams const& params) {
  if (params.count <= 0) {
    return;
  }
  queue.parallel_for<DeviceGuardKernel>(
      sycl::range<1>(static_cast<std::size_t>(params.count)), DeviceGuardKernel{params});
}

template <typename scalar_t>
struct ScatterRowsParams {
  scalar_t* dst = nullptr;
  scalar_t const* intermediate = nullptr;
  int64_t const* slots = nullptr;
  int64_t const* steps = nullptr;
  int main_count = 0;
  int t_max = 0;
  int row_elems = 0;
};

template <typename scalar_t>
struct ScatterRowsScalarKernel {
  ScatterRowsParams<scalar_t> p;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t total_lanes = static_cast<int64_t>(p.main_count) * p.row_elems;
    const int64_t lane = static_cast<int64_t>(item.get_global_id(0));
    if (lane >= total_lanes) {
      return;
    }
    const int request = static_cast<int>(lane / p.row_elems);
    const int elem = static_cast<int>(lane - static_cast<int64_t>(request) * p.row_elems);
    const int64_t step = p.steps[request];
    if (step < 0) {
      return;
    }

    const int64_t slot = p.slots[request];
    const int64_t dst_base = slot * p.row_elems;
    const int64_t src_base = (slot * p.t_max + step) * p.row_elems;
    p.dst[dst_base + elem] = p.intermediate[src_base + elem];
  }
};

template <typename scalar_t>
struct ScatterRowsPackKernel {
  using Pack = sycl::vec<uint32_t, kCopyPackBytes / static_cast<int>(sizeof(uint32_t))>;
  static constexpr int kPackElems = kCopyPackBytes / static_cast<int>(sizeof(scalar_t));

  ScatterRowsParams<scalar_t> p;

  void operator()(sycl::nd_item<1> item) const {
    const int packs_per_row = p.row_elems / kPackElems;
    const int64_t total_lanes = static_cast<int64_t>(p.main_count) * packs_per_row;
    const int64_t lane = static_cast<int64_t>(item.get_global_id(0));
    if (lane >= total_lanes) {
      return;
    }
    const int request = static_cast<int>(lane / packs_per_row);
    const int pack = static_cast<int>(lane - static_cast<int64_t>(request) * packs_per_row);
    const int64_t step = p.steps[request];
    if (step < 0) {
      return;
    }

    const int64_t slot = p.slots[request];
    const int64_t dst_base = slot * p.row_elems;
    const int64_t src_base = (slot * p.t_max + step) * p.row_elems;
    auto* dst_pack = reinterpret_cast<Pack*>(p.dst + dst_base);
    auto const* src_pack = reinterpret_cast<Pack const*>(p.intermediate + src_base);
    dst_pack[pack] = src_pack[pack];
  }
};

template <typename scalar_t>
bool can_use_pack_path(ScatterRowsParams<scalar_t> const& params) {
  constexpr int kPackElems = kCopyPackBytes / static_cast<int>(sizeof(scalar_t));
  auto aligned = [](void const* ptr) {
    return (reinterpret_cast<std::uintptr_t>(ptr) % kCopyPackBytes) == 0;
  };
  return params.row_elems % kPackElems == 0 && aligned(params.dst) && aligned(params.intermediate);
}

template <typename Kernel>
void submit_1d(sycl::queue& queue, int64_t lanes, Kernel const& kernel) {
  if (lanes <= 0) {
    return;
  }
  constexpr int64_t kLocal = 64;
  TORCH_CHECK(lanes <= std::numeric_limits<int>::max(), "scatter launch grid is too large");
  const int64_t global = ceil_div_i64(lanes, kLocal) * kLocal;
  queue.parallel_for<Kernel>(
      sycl::nd_range<1>(
          sycl::range<1>(static_cast<std::size_t>(global)),
          sycl::range<1>(static_cast<std::size_t>(kLocal))),
      kernel);
}

template <typename scalar_t>
void launch_scatter_rows_pass(sycl::queue& queue, ScatterRowsParams<scalar_t> const& params) {
  if (params.main_count <= 0 || params.row_elems <= 0) {
    return;
  }
  if (can_use_pack_path(params)) {
    constexpr int kPackElems = kCopyPackBytes / static_cast<int>(sizeof(scalar_t));
    submit_1d(queue, static_cast<int64_t>(params.main_count) * (params.row_elems / kPackElems),
              ScatterRowsPackKernel<scalar_t>{params});
    return;
  }
  submit_1d(queue, static_cast<int64_t>(params.main_count) * params.row_elems, ScatterRowsScalarKernel<scalar_t>{params});
}

template <typename scalar_t>
void launch_scatter_rows(
    sycl::queue& queue,
    scalar_t* dst,
    scalar_t const* intermediate,
    int row_elems,
    int t_max,
    int64_t const* slots,
    int64_t const* steps,
    int main_count,
    int64_t const* track_slots,
    int64_t const* track_steps,
    int track_count) {
  ScatterRowsParams<scalar_t> main_params{dst, intermediate, slots, steps, main_count, t_max, row_elems};
  launch_scatter_rows_pass(queue, main_params);
  ScatterRowsParams<scalar_t> track_params{dst, intermediate, track_slots, track_steps, track_count, t_max, row_elems};
  launch_scatter_rows_pass(queue, track_params);
}

int row_elems_from_state(at::Tensor const& state, char const* name) {
  CHECK_INPUT(state);
  TORCH_CHECK(state.dim() >= 2, name, " must have at least 2 dimensions");
  const int64_t slots = state.size(0);
  TORCH_CHECK(slots > 0, name, " slot dimension must be positive");
  TORCH_CHECK(state.numel() % slots == 0, name, " first dimension must divide numel");
  return checked_int64_to_int(state.numel() / slots, name);
}

void check_intermediate_shape(
    at::Tensor const& intermediate,
    at::Tensor const& state,
    int t_max,
    int row_elems,
    char const* name) {
  CHECK_INPUT(intermediate);
  TORCH_CHECK(intermediate.scalar_type() == state.scalar_type(), name, " dtype must match state dtype");
  TORCH_CHECK(intermediate.size(0) == state.size(0), name, " slot dimension must match state");
  TORCH_CHECK(
      intermediate.numel() == state.size(0) * static_cast<int64_t>(t_max) * row_elems,
      name,
      " must contain slots * t_max * row_elems elements");
}

void check_index_vector(at::Tensor const& tensor, int64_t expected, char const* name) {
  CHECK_INPUT(tensor);
  TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Long, name, " must be int64");
  TORCH_CHECK(tensor.dim() == 1, name, " must be 1D");
  TORCH_CHECK(tensor.numel() == expected, name, " has unexpected length");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> inkling_dflash_cache_path(
    const at::Tensor& req_to_token,
    const at::Tensor& req_pool_indices,
    const at::Tensor& pos2d,
    const at::Tensor& mask,
    const at::Tensor& out_offsets,
    int64_t gather_count,
    const at::Tensor& logits) {
  CHECK_INPUT(req_to_token);
  CHECK_INPUT(req_pool_indices);
  CHECK_INPUT(pos2d);
  CHECK_INPUT(mask);
  CHECK_INPUT(out_offsets);
  CHECK_INPUT(logits);
  TORCH_CHECK(req_to_token.scalar_type() == at::ScalarType::Long, "req_to_token must be int64");
  TORCH_CHECK(req_pool_indices.scalar_type() == at::ScalarType::Long, "req_pool_indices must be int64");
  TORCH_CHECK(pos2d.scalar_type() == at::ScalarType::Long, "pos2d must be int64");
  TORCH_CHECK(mask.scalar_type() == at::ScalarType::Byte, "mask must be uint8");
  TORCH_CHECK(out_offsets.scalar_type() == at::ScalarType::Int, "out_offsets must be int32");
  TORCH_CHECK(logits.scalar_type() == at::ScalarType::Float, "logits must be float32");
  TORCH_CHECK(req_to_token.dim() == 2, "req_to_token must be 2D");
  TORCH_CHECK(req_pool_indices.dim() == 1, "req_pool_indices must be 1D");
  TORCH_CHECK(pos2d.dim() == 2, "pos2d must be 2D [batch, draft_tokens]");
  TORCH_CHECK(mask.sizes() == pos2d.sizes(), "mask shape must match pos2d");
  TORCH_CHECK(out_offsets.sizes() == pos2d.sizes(), "out_offsets shape must match pos2d");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D [tokens, vocab]");
  TORCH_CHECK(req_pool_indices.numel() == pos2d.size(0), "req_pool_indices length must match batch");

  const int batch = checked_int64_to_int(pos2d.size(0), "batch", true);
  const int draft_tokens = checked_int64_to_int(pos2d.size(1), "draft_tokens", true);
  const int table_width = checked_int64_to_int(req_to_token.size(1), "table_width");
  const int tokens = checked_int64_to_int(logits.size(0), "tokens", true);
  const int vocab = checked_int64_to_int(logits.size(1), "vocab");
  const int out_count = checked_int64_to_int(gather_count, "gather_count", true);

  at::Tensor gathered = at::empty({out_count}, req_to_token.options());
  at::Tensor greedy_tokens = at::empty({tokens}, req_to_token.options());

  sycl::queue& queue = dpcppGetCurrentQueue();
  MaskedGatherParams gather_params;
  gather_params.req_to_token = req_to_token.data_ptr<int64_t>();
  gather_params.req_pool_indices = req_pool_indices.data_ptr<int64_t>();
  gather_params.pos2d = pos2d.data_ptr<int64_t>();
  gather_params.mask = mask.data_ptr<uint8_t>();
  gather_params.out_offsets = out_offsets.data_ptr<int32_t>();
  gather_params.out = gathered.data_ptr<int64_t>();
  gather_params.batch = batch;
  gather_params.draft_tokens = draft_tokens;
  gather_params.table_width = table_width;
  launch_masked_gather(queue, gather_params);

  LogitsArgmaxParams argmax_params;
  argmax_params.logits = logits.data_ptr<float>();
  argmax_params.out_tokens = greedy_tokens.data_ptr<int64_t>();
  argmax_params.tokens = tokens;
  argmax_params.vocab = vocab;
  launch_logits_argmax(queue, argmax_params);

  return {gathered, greedy_tokens};
}

std::tuple<at::Tensor, at::Tensor> inkling_dflash_device_guard(const at::Tensor& names) {
  CHECK_INPUT(names);
  TORCH_CHECK(names.scalar_type() == at::ScalarType::Byte, "names must be uint8 ASCII bytes");
  TORCH_CHECK(names.dim() == 2, "names must be 2D [count, stride]");
  const int count = checked_int64_to_int(names.size(0), "count", true);
  const int stride = checked_int64_to_int(names.size(1), "stride");
  TORCH_CHECK(stride >= kNameStrideMinimum, "names stride must be at least 10 bytes");

  at::Tensor legacy_cuda_guard = at::empty({count}, names.options());
  at::Tensor dflash_supported_guard = at::empty({count}, names.options());

  DeviceGuardParams params;
  params.names = reinterpret_cast<char const*>(names.data_ptr<uint8_t>());
  params.legacy_cuda_guard = legacy_cuda_guard.data_ptr<uint8_t>();
  params.dflash_supported_guard = dflash_supported_guard.data_ptr<uint8_t>();
  params.count = count;
  params.stride = stride;
  launch_device_guard(dpcppGetCurrentQueue(), params);

  return {legacy_cuda_guard, dflash_supported_guard};
}

void inkling_scatter_mamba_states_after_mtp_verify(
    at::Tensor& ssm_states,
    const at::Tensor& ssm_intermediate,
    at::Tensor& conv_a_states,
    const at::Tensor& conv_a_intermediate,
    at::Tensor& conv_b_states,
    const at::Tensor& conv_b_intermediate,
    const at::Tensor& slots,
    const at::Tensor& steps,
    const std::optional<at::Tensor>& track_slots,
    const std::optional<at::Tensor>& track_steps,
    int64_t t_max) {
  const int t_max_i = checked_int64_to_int(t_max, "t_max");
  const int main_count = checked_int64_to_int(slots.numel(), "main_count", true);
  check_index_vector(slots, main_count, "slots");
  check_index_vector(steps, main_count, "steps");

  TORCH_CHECK(track_slots.has_value() == track_steps.has_value(), "track_slots and track_steps must be both set or both None");
  int track_count = 0;
  int64_t const* track_slots_ptr = nullptr;
  int64_t const* track_steps_ptr = nullptr;
  if (track_slots.has_value()) {
    track_count = checked_int64_to_int(track_slots->numel(), "track_count", true);
    check_index_vector(*track_slots, track_count, "track_slots");
    check_index_vector(*track_steps, track_count, "track_steps");
    track_slots_ptr = track_slots->data_ptr<int64_t>();
    track_steps_ptr = track_steps->data_ptr<int64_t>();
  }

  TORCH_CHECK(conv_a_states.scalar_type() == ssm_states.scalar_type(), "conv_a_states dtype must match ssm_states");
  TORCH_CHECK(conv_b_states.scalar_type() == ssm_states.scalar_type(), "conv_b_states dtype must match ssm_states");
  const int ssm_row_elems = row_elems_from_state(ssm_states, "ssm_states");
  const int conv_a_row_elems = row_elems_from_state(conv_a_states, "conv_a_states");
  const int conv_b_row_elems = row_elems_from_state(conv_b_states, "conv_b_states");
  TORCH_CHECK(conv_a_states.size(0) == ssm_states.size(0), "conv_a_states slot dimension must match ssm_states");
  TORCH_CHECK(conv_b_states.size(0) == ssm_states.size(0), "conv_b_states slot dimension must match ssm_states");
  check_intermediate_shape(ssm_intermediate, ssm_states, t_max_i, ssm_row_elems, "ssm_intermediate");
  check_intermediate_shape(conv_a_intermediate, conv_a_states, t_max_i, conv_a_row_elems, "conv_a_intermediate");
  check_intermediate_shape(conv_b_intermediate, conv_b_states, t_max_i, conv_b_row_elems, "conv_b_intermediate");

  sycl::queue& queue = dpcppGetCurrentQueue();
  DISPATCH_FLOAT_TYPES(ssm_states.scalar_type(), "inkling_scatter_mamba_states_after_mtp_verify", [&] {
    launch_scatter_rows<scalar_t>(
        queue,
        ssm_states.data_ptr<scalar_t>(),
        ssm_intermediate.data_ptr<scalar_t>(),
        ssm_row_elems,
        t_max_i,
        slots.data_ptr<int64_t>(),
        steps.data_ptr<int64_t>(),
        main_count,
        track_slots_ptr,
        track_steps_ptr,
        track_count);
    launch_scatter_rows<scalar_t>(
        queue,
        conv_a_states.data_ptr<scalar_t>(),
        conv_a_intermediate.data_ptr<scalar_t>(),
        conv_a_row_elems,
        t_max_i,
        slots.data_ptr<int64_t>(),
        steps.data_ptr<int64_t>(),
        main_count,
        track_slots_ptr,
        track_steps_ptr,
        track_count);
    launch_scatter_rows<scalar_t>(
        queue,
        conv_b_states.data_ptr<scalar_t>(),
        conv_b_intermediate.data_ptr<scalar_t>(),
        conv_b_row_elems,
        t_max_i,
        slots.data_ptr<int64_t>(),
        steps.data_ptr<int64_t>(),
        main_count,
        track_slots_ptr,
        track_steps_ptr,
        track_count);
  });
}
