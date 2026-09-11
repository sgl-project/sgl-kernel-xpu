/* Copyright 2026 SGLang Team. All Rights Reserved.
 * Copyright (C) 2026 Intel Corporation, All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 */

#include <ATen/ATen.h>
#include <torch/all.h>

#include <algorithm>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "comm/Numerics.h"
#include "sgl_kernel_export.h"

namespace {

using bf16_t = sycl::ext::oneapi::bfloat16;

constexpr int64_t kDefaultBlock = 256;
constexpr int64_t kRelProjVec = 8;

// Production Inkling shapes: r is [t, h, 16] and proj is [16, 1024].
constexpr int64_t kRelProjD = 16;
constexpr int64_t kRelProjE = 1024;

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

struct RelProjParams {
  const bf16_t* r = nullptr;
  const bf16_t* proj = nullptr;
  const float* tau = nullptr;
  bf16_t* out = nullptr;
  int64_t t = 0;
  int64_t h = 0;
  int64_t r_stride_t = 0;
};

// out[t, h, :] = bf16(tau[t] * r[t, h, :]) @ proj
//
// The projection matrix is only 16x1024 (32 KiB) while the row count t*h is 6 to
// 768, so the arithmetic is trivial and the kernel is bound by how many times
// `proj` is re-read. One work-item owns `Vec` consecutive output columns and
// keeps that column slice of `proj` in registers, then walks `MTile` rows of
// r: `proj` is fetched once per (row tile, column slice) instead of once per
// output row. `MTile` trades that reuse against the number of work-items, which
// is what keeps the tiny decode shapes latency-bound rather than starved.
template <int MTile, int Vec>
class InklingRelProjKernel {
 public:
  static_assert(Vec % 2 == 0, "Vec must be even so proj/out move as 32-bit pairs");

  RelProjParams params;
  int64_t total;
  int64_t col_slices;

  void operator()(sycl::nd_item<1> item) const {
    int64_t idx = static_cast<int64_t>(item.get_global_id(0));
    if (idx >= total) {
      return;
    }

    // Consecutive work-items take consecutive column slices so that a subgroup
    // reads and writes one contiguous run of `proj` / `out`.
    int64_t col_slice = idx % col_slices;
    int64_t m_tile = idx / col_slices;
    int64_t e0 = col_slice * Vec;

    float proj_tile[kRelProjD][Vec];
#pragma unroll
    for (int64_t d = 0; d < kRelProjD; ++d) {
      const uint32_t* proj_row = reinterpret_cast<const uint32_t*>(params.proj + d * kRelProjE);
#pragma unroll
      for (int64_t i = 0; i < Vec / 2; ++i) {
        uint32_t pair = proj_row[e0 / 2 + i];
        proj_tile[d][2 * i] = bf16_raw_to_float(static_cast<uint16_t>(pair & 0xffffu));
        proj_tile[d][2 * i + 1] = bf16_raw_to_float(static_cast<uint16_t>(pair >> 16));
      }
    }

    int64_t m = m_tile * MTile;
    int64_t rows = params.t * params.h;
    int64_t ti = m / params.h;
    int64_t hi = m - ti * params.h;

#pragma unroll
    for (int64_t mm = 0; mm < MTile; ++mm) {
      if (m >= rows) {
        return;
      }

      float scale = params.tau[ti];
      const bf16_t* r_row = params.r + ti * params.r_stride_t + hi * kRelProjD;

      float acc[Vec];
#pragma unroll
      for (int64_t i = 0; i < Vec; ++i) {
        acc[i] = 0.0f;
      }

#pragma unroll
      for (int64_t d = 0; d < kRelProjD; ++d) {
        // Pre-scaling r to bf16 before the fp32 accumulation is part of the
        // reference numerics, not an approximation.
        float r_value = bf16_to_float(float_to_bf16(bf16_to_float(r_row[d]) * scale));
#pragma unroll
        for (int64_t i = 0; i < Vec; ++i) {
          acc[i] += r_value * proj_tile[d][i];
        }
      }

      uint32_t* out_row = reinterpret_cast<uint32_t*>(params.out + m * kRelProjE);
#pragma unroll
      for (int64_t i = 0; i < Vec / 2; ++i) {
        out_row[e0 / 2 + i] = static_cast<uint32_t>(bf16_float_to_raw(acc[2 * i])) |
                              (static_cast<uint32_t>(bf16_float_to_raw(acc[2 * i + 1])) << 16);
      }

      ++m;
      if (++hi == params.h) {
        hi = 0;
        ++ti;
      }
    }
  }
};

template <int MTile, int Vec>
void submit_rel_proj_kernel(sycl::queue& queue, const RelProjParams& params) {
  int64_t col_slices = kRelProjE / Vec;
  int64_t total = CeilDiv(params.t * params.h, static_cast<int64_t>(MTile)) * col_slices;
  int64_t block = std::min<int64_t>(kDefaultBlock, col_slices);
  int64_t global = RoundUp(total, block);
  InklingRelProjKernel<MTile, Vec> kernel{params, total, col_slices};
  sycl_kernel_submit(global, block, queue, kernel);
}

void launch_rel_proj_kernel(sycl::queue& queue, const RelProjParams& params) {
  int64_t rows = params.t * params.h;
  if (rows == 0) {
    return;
  }

  // Deepen the row tile only while enough row tiles remain to fill the machine:
  // below ~kMinRowTiles tiles the kernel is latency-bound and extra `proj` reuse
  // costs more parallelism than it saves bandwidth. Tuned on BMG/B60.
  constexpr int64_t kMinRowTiles = 40;
  if (rows >= kMinRowTiles * 16) {
    submit_rel_proj_kernel<16, kRelProjVec>(queue, params);
  } else if (rows >= kMinRowTiles * 8) {
    submit_rel_proj_kernel<8, kRelProjVec>(queue, params);
  } else if (rows >= kMinRowTiles * 4) {
    submit_rel_proj_kernel<4, kRelProjVec>(queue, params);
  } else if (rows >= kMinRowTiles * 2) {
    submit_rel_proj_kernel<2, kRelProjVec>(queue, params);
  } else {
    submit_rel_proj_kernel<1, kRelProjVec>(queue, params);
  }
}

}  // namespace

SGL_KERNEL_EXPORT at::Tensor
inkling_rel_proj_small_t(const at::Tensor& r, const at::Tensor& proj, const at::Tensor& tau, const at::Tensor& out) {
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
  TORCH_CHECK(r.size(2) == kRelProjD, "inkling_rel_proj_small_t: only production d_rel=16 is supported");
  TORCH_CHECK(proj.size(1) == kRelProjE, "inkling_rel_proj_small_t: only production rel_extent=1024 is supported");
  TORCH_CHECK(
      r.size(0) == 1 || r.stride(0) > r.size(1) * r.size(2),
      "inkling_rel_proj_small_t: r must be the strided trailing view of the packed qkvr output");
  TORCH_CHECK(proj.is_contiguous(), "inkling_rel_proj_small_t: proj must be contiguous");
  TORCH_CHECK(out.is_contiguous(), "inkling_rel_proj_small_t: out must be contiguous");

  CHECK_DEVICE(tau);
  TORCH_CHECK(tau.scalar_type() == at::ScalarType::Float, "inkling_rel_proj_small_t: tau must be float32");
  TORCH_CHECK(tau.dim() == 1, "inkling_rel_proj_small_t: tau must be 1D");
  TORCH_CHECK(tau.numel() == r.size(0), "inkling_rel_proj_small_t: tau must have one entry per token");
  TORCH_CHECK(tau.is_contiguous(), "inkling_rel_proj_small_t: tau must be contiguous");

  RelProjParams params{};
  params.r = reinterpret_cast<const bf16_t*>(r.data_ptr<at::BFloat16>());
  params.proj = reinterpret_cast<const bf16_t*>(proj.data_ptr<at::BFloat16>());
  params.tau = tau.data_ptr<float>();
  params.out = reinterpret_cast<bf16_t*>(out.data_ptr<at::BFloat16>());
  params.t = r.size(0);
  params.h = r.size(1);
  params.r_stride_t = r.stride(0);

  // The kernel moves proj and out as bf16 pairs.
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(params.proj) % sizeof(uint32_t) == 0,
      "inkling_rel_proj_small_t: proj must be 4-byte aligned");
  TORCH_CHECK(
      reinterpret_cast<uintptr_t>(params.out) % sizeof(uint32_t) == 0,
      "inkling_rel_proj_small_t: out must be 4-byte aligned");

  auto queue = dpcppGetCurrentQueue();
  launch_rel_proj_kernel(queue, params);
  return out;
}
