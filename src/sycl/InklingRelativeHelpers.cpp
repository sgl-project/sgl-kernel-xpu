/* Copyright 2026 SGLang Team. All Rights Reserved.
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <ATen/ATen.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "Utils.h"

namespace {

using bf16_t = sycl::ext::oneapi::bfloat16;

constexpr int64_t kDefaultBlock = 256;
constexpr int64_t kRowPackBytes = 16;
constexpr int kRowPackWords = kRowPackBytes / static_cast<int>(sizeof(uint32_t));
constexpr int64_t kRowPackElems = kRowPackBytes / static_cast<int64_t>(sizeof(bf16_t));
constexpr int64_t kRowSmallLaneElems = kRowPackElems;
constexpr int64_t kRowLargePacksPerLane = 4;
constexpr int64_t kRowLargeLaneElems = kRowPackElems * kRowLargePacksPerLane;
constexpr int kRowEsimdCopyWords = 64;
constexpr int64_t kRowEsimdMinRows = 512;
constexpr int64_t kRowSmallWorkItemsThreshold = 8192;
constexpr int64_t kRelProjVec = 8;

inline int64_t ceil_div_i64(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

inline int64_t round_up_i64(int64_t x, int64_t multiple) {
  return ceil_div_i64(x, multiple) * multiple;
}

inline bool is_aligned(const void* ptr, int64_t alignment) {
  return reinterpret_cast<std::uintptr_t>(ptr) % static_cast<std::uintptr_t>(alignment) == 0;
}

inline float bf16_raw_to_float(uint16_t raw) {
  return sycl::bit_cast<float>(static_cast<uint32_t>(raw) << 16);
}

inline uint16_t bf16_float_to_raw(float value) {
  uint32_t bits = sycl::bit_cast<uint32_t>(value);
  uint32_t lsb = (bits >> 16) & 1u;
  uint32_t rounding_bias = 0x7fffu + lsb;
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

inline float bf16_to_float(bf16_t value) {
  return bf16_raw_to_float(sycl::bit_cast<uint16_t>(value));
}

inline bf16_t float_to_bf16(float value) {
  return sycl::bit_cast<bf16_t>(bf16_float_to_raw(value));
}

inline uint64_t scale_bf16_pack4(uint64_t raw, float scale) {
  uint64_t out = 0;
#pragma unroll
  for (int lane = 0; lane < 4; ++lane) {
    uint16_t in_bits = static_cast<uint16_t>(raw >> (16 * lane));
    uint16_t out_bits = bf16_float_to_raw(bf16_raw_to_float(in_bits) * scale);
    out |= static_cast<uint64_t>(out_bits) << (16 * lane);
  }
  return out;
}

struct RowParams {
  const bf16_t* x = nullptr;
  const float* tau = nullptr;
  bf16_t* out = nullptr;
  int64_t rows = 0;
  int64_t inner = 0;
  int64_t stride = 0;
  int64_t lanes_per_row = 0;
  int64_t vec_count = 0;
};

class InklingRowCompactEsimdKernel {
 public:
  RowParams params;
  int64_t chunks_per_row;

  void operator()(sycl::item<1> item) const SYCL_ESIMD_KERNEL {
    int64_t linear = static_cast<int64_t>(item.get_linear_id());
    int64_t total = params.rows * chunks_per_row;
    if (linear >= total) {
      return;
    }
    int64_t row = linear / chunks_per_row;
    int64_t chunk = linear - row * chunks_per_row;
    const bf16_t* src_row = params.x + row * params.stride;
    bf16_t* dst_row = params.out + row * params.inner;
    auto value = sycl::ext::intel::esimd::block_load<uint32_t, kRowEsimdCopyWords>(
        reinterpret_cast<const uint32_t*>(src_row) + chunk * kRowEsimdCopyWords);
    sycl::ext::intel::esimd::block_store<uint32_t, kRowEsimdCopyWords>(
        reinterpret_cast<uint32_t*>(dst_row) + chunk * kRowEsimdCopyWords, value);
  }
};

class InklingRowScaleBf16EsimdKernel {
 public:
  RowParams params;
  int64_t chunks_per_row;

  void operator()(sycl::item<1> item) const SYCL_ESIMD_KERNEL {
    int64_t linear = static_cast<int64_t>(item.get_linear_id());
    int64_t total = params.rows * chunks_per_row;
    if (linear >= total) {
      return;
    }
    int64_t row = linear / chunks_per_row;
    int64_t chunk = linear - row * chunks_per_row;
    const bf16_t* src_row = params.x + row * params.stride;
    bf16_t* dst_row = params.out + row * params.inner;
    float scale = params.tau[row];

    auto raw = sycl::ext::intel::esimd::block_load<uint32_t, kRowEsimdCopyWords>(
        reinterpret_cast<const uint32_t*>(src_row) + chunk * kRowEsimdCopyWords);
    auto lo_bits = (raw & 0x0000ffffu) << 16;
    auto hi_bits = raw & 0xffff0000u;
    auto lo = lo_bits.template bit_cast_view<float>();
    auto hi = hi_bits.template bit_cast_view<float>();
    lo = lo * scale;
    hi = hi * scale;

    auto lo_fbits = lo.template bit_cast_view<uint32_t>();
    auto hi_fbits = hi.template bit_cast_view<uint32_t>();
    auto lo_round = ((lo_fbits >> 16) & 1u) + 0x7fffu;
    auto hi_round = ((hi_fbits >> 16) & 1u) + 0x7fffu;
    auto out = ((lo_fbits + lo_round) >> 16) | (((hi_fbits + hi_round) >> 16) << 16);
    sycl::ext::intel::esimd::block_store<uint32_t, kRowEsimdCopyWords>(
        reinterpret_cast<uint32_t*>(dst_row) + chunk * kRowEsimdCopyWords, out);
  }
};

inline void launch_row_compact_esimd(sycl::queue& queue, const RowParams& params, int64_t chunks_per_row) {
  int64_t total = params.rows * chunks_per_row;
  InklingRowCompactEsimdKernel kernel{params, chunks_per_row};
  queue.parallel_for<InklingRowCompactEsimdKernel>(sycl::range<1>(static_cast<std::size_t>(total)), kernel);
}

inline void launch_row_scale_bf16_esimd(sycl::queue& queue, const RowParams& params, int64_t chunks_per_row) {
  int64_t total = params.rows * chunks_per_row;
  InklingRowScaleBf16EsimdKernel kernel{params, chunks_per_row};
  queue.parallel_for<InklingRowScaleBf16EsimdKernel>(sycl::range<1>(static_cast<std::size_t>(total)), kernel);
}

template <bool HasTau, int64_t Vec>
class InklingRowKernel {
 public:
  RowParams params;
  int64_t total;

  void operator()(sycl::nd_item<1> item) const {
    constexpr int64_t kPacksPerLane = Vec / kRowPackElems;
    int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    if (idx >= total) {
      return;
    }
    int64_t row = idx / params.lanes_per_row;
    int64_t lane = idx - row * params.lanes_per_row;
    const bf16_t* src_row = params.x + row * params.stride;
    bf16_t* dst_row = params.out + row * params.inner;
    float scale = 1.0f;
    if constexpr (HasTau) {
      scale = params.tau[row];
    }

    if (lane < params.vec_count) {
      int64_t col0 = lane * Vec;
#pragma unroll
      for (int64_t pack = 0; pack < kPacksPerLane; ++pack) {
        const bf16_t* src_pack = src_row + col0 + pack * kRowPackElems;
        bf16_t* dst_pack = dst_row + col0 + pack * kRowPackElems;
        uint64_t raw0 = *reinterpret_cast<const uint64_t*>(src_pack);
        uint64_t raw1 = *reinterpret_cast<const uint64_t*>(src_pack + 4);
        if constexpr (HasTau) {
          raw0 = scale_bf16_pack4(raw0, scale);
          raw1 = scale_bf16_pack4(raw1, scale);
        }
        *reinterpret_cast<uint64_t*>(dst_pack) = raw0;
        *reinterpret_cast<uint64_t*>(dst_pack + 4) = raw1;
      }
      return;
    }

    int64_t col = params.vec_count * Vec + (lane - params.vec_count);
    bf16_t value = src_row[col];
    if constexpr (HasTau) {
      value = float_to_bf16(bf16_to_float(value) * scale);
    }
    dst_row[col] = value;
  }
};

template <bool HasTau, int64_t Vec>
void launch_row_kernel_static(sycl::queue& queue, RowParams params) {
  static_assert(Vec % kRowPackElems == 0);

  bool aligned = params.inner % kRowPackElems == 0 && params.stride % kRowPackElems == 0 &&
      is_aligned(params.x, kRowPackBytes) && is_aligned(params.out, kRowPackBytes);
  int64_t row_bytes = params.inner * static_cast<int64_t>(sizeof(bf16_t));
  int64_t row_words = row_bytes / static_cast<int64_t>(sizeof(uint32_t));
  if constexpr (!HasTau) {
    if (params.rows >= kRowEsimdMinRows && aligned && row_bytes % static_cast<int64_t>(sizeof(uint32_t)) == 0 &&
        row_words % kRowEsimdCopyWords == 0) {
      launch_row_compact_esimd(queue, params, row_words / kRowEsimdCopyWords);
      return;
    }
  }
  if constexpr (HasTau) {
    if (params.rows >= kRowEsimdMinRows && aligned && row_bytes % static_cast<int64_t>(sizeof(uint32_t)) == 0 &&
        row_words % kRowEsimdCopyWords == 0) {
      launch_row_scale_bf16_esimd(queue, params, row_words / kRowEsimdCopyWords);
      return;
    }
  }

  params.vec_count = aligned ? params.inner / Vec : 0;
  int64_t scalar_tail = params.inner - params.vec_count * Vec;
  params.lanes_per_row = params.vec_count + scalar_tail;
  int64_t total = params.rows * params.lanes_per_row;
  if (total == 0) {
    return;
  }

  int64_t global = round_up_i64(total, kDefaultBlock);
  InklingRowKernel<HasTau, Vec> kernel{params, total};
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(kDefaultBlock))),
        kernel);
  });
}

template <bool HasTau>
void launch_row_kernel(sycl::queue& queue, RowParams const& params) {
  int64_t logical_work_items = params.rows * ceil_div_i64(params.inner, kRowPackElems);
  if (logical_work_items <= kRowSmallWorkItemsThreshold) {
    launch_row_kernel_static<HasTau, kRowSmallLaneElems>(queue, params);
  } else {
    launch_row_kernel_static<HasTau, kRowLargeLaneElems>(queue, params);
  }
}

struct RelProjParams {
  const bf16_t* r = nullptr;
  const bf16_t* proj = nullptr;
  const float* tau = nullptr;
  bf16_t* out = nullptr;
  int64_t t = 0;
  int64_t h = 0;
  int64_t d = 0;
  int64_t e = 0;
  int64_t r_stride_t = 0;
};

template <bool HasTau>
class InklingRelProjBf16D16EsimdKernel {
 public:
  RelProjParams params;
  int64_t chunks_per_row;

  void operator()(sycl::item<1> item) const SYCL_ESIMD_KERNEL {
    int64_t linear = static_cast<int64_t>(item.get_linear_id());
    int64_t total = params.t * params.h * chunks_per_row;
    if (linear >= total) {
      return;
    }
    int64_t chunk = linear % chunks_per_row;
    int64_t th = linear / chunks_per_row;
    int64_t ti = th / params.h;
    int64_t hi = th - ti * params.h;
    int64_t e0 = chunk * 16;

    float scale = 1.0f;
    if constexpr (HasTau) {
      scale = params.tau[ti];
    }

    sycl::ext::intel::esimd::simd<float, 8> acc_lo(0.0f);
    sycl::ext::intel::esimd::simd<float, 8> acc_hi(0.0f);
    const bf16_t* r_row = params.r + ti * params.r_stride_t + hi * params.d;

#pragma unroll
    for (int d = 0; d < 16; ++d) {
      float rv = bf16_to_float(r_row[d]);
      if constexpr (HasTau) {
        rv = bf16_to_float(float_to_bf16(rv * scale));
      }
      auto raw = sycl::ext::intel::esimd::block_load<uint32_t, 8>(
          reinterpret_cast<const uint32_t*>(params.proj + static_cast<int64_t>(d) * params.e) + e0 / 2);
      auto lo_bits = (raw & 0x0000ffffu) << 16;
      auto hi_bits = raw & 0xffff0000u;
      acc_lo += lo_bits.template bit_cast_view<float>() * rv;
      acc_hi += hi_bits.template bit_cast_view<float>() * rv;
    }

    auto lo_fbits = acc_lo.template bit_cast_view<uint32_t>();
    auto hi_fbits = acc_hi.template bit_cast_view<uint32_t>();
    auto lo_round = ((lo_fbits >> 16) & 1u) + 0x7fffu;
    auto hi_round = ((hi_fbits >> 16) & 1u) + 0x7fffu;
    auto out = ((lo_fbits + lo_round) >> 16) | (((hi_fbits + hi_round) >> 16) << 16);
    bf16_t* out_row = params.out + (ti * params.h + hi) * params.e;
    sycl::ext::intel::esimd::block_store<uint32_t, 8>(reinterpret_cast<uint32_t*>(out_row) + e0 / 2, out);
  }
};

template <bool HasTau>
inline void launch_rel_proj_bf16_d16_esimd(sycl::queue& queue, const RelProjParams& params) {
  int64_t chunks_per_row = params.e / 16;
  int64_t total = params.t * params.h * chunks_per_row;
  InklingRelProjBf16D16EsimdKernel<HasTau> kernel{params, chunks_per_row};
  queue.parallel_for<InklingRelProjBf16D16EsimdKernel<HasTau>>(
      sycl::range<1>(static_cast<std::size_t>(total)), kernel);
}

template <bool HasTau, int64_t Vec>
class InklingRelProjKernel {
 public:
  RelProjParams params;
  int64_t total;
  int64_t e_vecs;

  void operator()(sycl::nd_item<1> item) const {
    int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    if (idx >= total) {
      return;
    }

    int64_t ev = idx % e_vecs;
    int64_t th = idx / e_vecs;
    int64_t ti = th / params.h;
    int64_t hi = th - ti * params.h;
    int64_t e0 = ev * Vec;

    float scale = 1.0f;
    if constexpr (HasTau) {
      scale = params.tau[ti];
    }

    float acc[Vec];
#pragma unroll
    for (int64_t i = 0; i < Vec; ++i) {
      acc[i] = 0.0f;
    }

    const bf16_t* r_row = params.r + ti * params.r_stride_t + hi * params.d;
    for (int64_t d = 0; d < params.d; ++d) {
      float rv = bf16_to_float(r_row[d]);
      if constexpr (HasTau) {
        rv = bf16_to_float(float_to_bf16(rv * scale));
      }
      const bf16_t* proj_row = params.proj + d * params.e;
#pragma unroll
      for (int64_t i = 0; i < Vec; ++i) {
        int64_t e_col = e0 + i;
        if (e_col < params.e) {
          acc[i] += rv * bf16_to_float(proj_row[e_col]);
        }
      }
    }

    bf16_t* out_row = params.out + (ti * params.h + hi) * params.e;
#pragma unroll
    for (int64_t i = 0; i < Vec; ++i) {
      int64_t e_col = e0 + i;
      if (e_col < params.e) {
        out_row[e_col] = float_to_bf16(acc[i]);
      }
    }
  }
};

template <bool HasTau, int64_t Vec = kRelProjVec>
void launch_rel_proj_kernel(sycl::queue& queue, RelProjParams params) {
  int64_t e_vecs = ceil_div_i64(params.e, Vec);
  int64_t total = params.t * params.h * e_vecs;
  if (total == 0) {
    return;
  }

  if (params.t <= 4 && params.d == 16 && params.e % 16 == 0) {
    launch_rel_proj_bf16_d16_esimd<HasTau>(queue, params);
    return;
  }

  int64_t global = round_up_i64(total, kDefaultBlock);
  InklingRelProjKernel<HasTau, Vec> kernel{params, total, e_vecs};
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(static_cast<std::size_t>(global)),
            sycl::range<1>(static_cast<std::size_t>(kDefaultBlock))),
        kernel);
  });
}

void check_row_inputs(const at::Tensor& x, const at::Tensor& out, const char* op_name) {
  CHECK_DEVICE(x);
  CHECK_DEVICE(out);
  TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16, op_name, ": x must be bfloat16");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, op_name, ": out must be bfloat16");
  TORCH_CHECK(x.dim() == 2, op_name, ": x must have shape [rows, inner]");
  TORCH_CHECK(out.sizes() == x.sizes(), op_name, ": out must have the same shape as x");
  TORCH_CHECK(x.stride(1) == 1, op_name, ": x must be contiguous on the inner dimension");
  TORCH_CHECK(out.is_contiguous(), op_name, ": out must be contiguous");
}

}  // namespace

at::Tensor inkling_row_scale_bf16(const at::Tensor& x, const at::Tensor& tau, const at::Tensor& out) {
  check_row_inputs(x, out, "inkling_row_scale_bf16");
  CHECK_DEVICE(tau);
  TORCH_CHECK(tau.scalar_type() == at::ScalarType::Float, "inkling_row_scale_bf16: tau must be float32");
  TORCH_CHECK(tau.dim() == 1, "inkling_row_scale_bf16: tau must be 1D");
  TORCH_CHECK(tau.numel() == x.size(0), "inkling_row_scale_bf16: tau must have one entry per row");
  TORCH_CHECK(tau.is_contiguous(), "inkling_row_scale_bf16: tau must be contiguous");

  RowParams params{};
  params.x = reinterpret_cast<const bf16_t*>(x.data_ptr<at::BFloat16>());
  params.tau = tau.data_ptr<float>();
  params.out = reinterpret_cast<bf16_t*>(out.data_ptr<at::BFloat16>());
  params.rows = x.size(0);
  params.inner = x.size(1);
  params.stride = x.stride(0);

  auto queue = dpcppGetCurrentQueue();
  launch_row_kernel<true>(queue, params);
  return out;
}

at::Tensor inkling_row_compact_bf16(const at::Tensor& x, const at::Tensor& out) {
  check_row_inputs(x, out, "inkling_row_compact_bf16");

  RowParams params{};
  params.x = reinterpret_cast<const bf16_t*>(x.data_ptr<at::BFloat16>());
  params.out = reinterpret_cast<bf16_t*>(out.data_ptr<at::BFloat16>());
  params.rows = x.size(0);
  params.inner = x.size(1);
  params.stride = x.stride(0);

  auto queue = dpcppGetCurrentQueue();
  launch_row_kernel<false>(queue, params);
  return out;
}

at::Tensor inkling_rel_proj_small_t(
    const at::Tensor& r,
    const at::Tensor& proj,
    const std::optional<at::Tensor>& tau,
    const at::Tensor& out) {
  CHECK_DEVICE(r);
  CHECK_DEVICE(proj);
  CHECK_DEVICE(out);
  TORCH_CHECK(r.scalar_type() == at::ScalarType::BFloat16, "inkling_rel_proj_small_t: r must be bfloat16");
  TORCH_CHECK(proj.scalar_type() == at::ScalarType::BFloat16, "inkling_rel_proj_small_t: proj must be bfloat16");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::BFloat16, "inkling_rel_proj_small_t: out must be bfloat16");
  TORCH_CHECK(r.dim() == 3, "inkling_rel_proj_small_t: r must have shape [t, h, d]");
  TORCH_CHECK(proj.dim() == 2, "inkling_rel_proj_small_t: proj must have shape [d, e]");
  TORCH_CHECK(out.dim() == 3, "inkling_rel_proj_small_t: out must have shape [t, h, e]");
  TORCH_CHECK(r.size(2) == proj.size(0), "inkling_rel_proj_small_t: r.size(2) must equal proj.size(0)");
  TORCH_CHECK(out.size(0) == r.size(0), "inkling_rel_proj_small_t: out.size(0) must equal r.size(0)");
  TORCH_CHECK(out.size(1) == r.size(1), "inkling_rel_proj_small_t: out.size(1) must equal r.size(1)");
  TORCH_CHECK(out.size(2) == proj.size(1), "inkling_rel_proj_small_t: out.size(2) must equal proj.size(1)");
  TORCH_CHECK(r.stride(2) == 1, "inkling_rel_proj_small_t: r must be contiguous on the d dimension");
  TORCH_CHECK(r.stride(1) == r.size(2), "inkling_rel_proj_small_t: r must have contiguous [h, d] rows");
  TORCH_CHECK(proj.is_contiguous(), "inkling_rel_proj_small_t: proj must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "inkling_rel_proj_small_t: out must be contiguous");

  const float* tau_ptr = nullptr;
  bool has_tau = tau.has_value() && tau->numel() > 0;
  if (has_tau) {
    const at::Tensor& tau_tensor = tau.value();
    CHECK_DEVICE(tau_tensor);
    TORCH_CHECK(tau_tensor.scalar_type() == at::ScalarType::Float, "inkling_rel_proj_small_t: tau must be float32");
    TORCH_CHECK(tau_tensor.dim() == 1, "inkling_rel_proj_small_t: tau must be 1D");
    TORCH_CHECK(tau_tensor.numel() == r.size(0), "inkling_rel_proj_small_t: tau must have one entry per token");
    TORCH_CHECK(tau_tensor.is_contiguous(), "inkling_rel_proj_small_t: tau must be contiguous");
    tau_ptr = tau_tensor.data_ptr<float>();
  }

  RelProjParams params{};
  params.r = reinterpret_cast<const bf16_t*>(r.data_ptr<at::BFloat16>());
  params.proj = reinterpret_cast<const bf16_t*>(proj.data_ptr<at::BFloat16>());
  params.tau = tau_ptr;
  params.out = reinterpret_cast<bf16_t*>(out.data_ptr<at::BFloat16>());
  params.t = r.size(0);
  params.h = r.size(1);
  params.d = r.size(2);
  params.e = proj.size(1);
  params.r_stride_t = r.stride(0);

  auto queue = dpcppGetCurrentQueue();
  if (has_tau) {
    launch_rel_proj_kernel<true>(queue, params);
  } else {
    launch_rel_proj_kernel<false>(queue, params);
  }
  return out;
}
