/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

// Per-token (per-row) dynamic FP8 E4M3 quantization for XPU.
// SYCL port of sgl-kernel csrc/gemm/per_token_quant_fp8.cu.
//
// For each row of a [num_tokens, hidden_dim] tensor:
//   scale = rowmax(|x|) / 448        (FP8_E4M3_MAX)
//   q[i]  = clamp(x[i] / scale, -448, 448)  as e4m3
// scale is written per token (row-major float32).

#include <ATen/ATen.h>
#include <cutlass/float8.h>

#include <sycl/sycl.hpp>

#include "MemoryAccess.h"
#include "SYCLHelpers.h"
#include "Utils.h"

// TODO: Remove CUTLASS emulation and use native SYCL FP8 when available.
using cutlass::float_e4m3_t;

static constexpr int SUB_GROUP_SIZE = 32;
static constexpr int TOKENS_PER_WG = 8;  // sub-groups (tokens) per work-group in the warp kernel
constexpr float FP8_E4M3_MAX = 448.0f;

// ---------------------------------------------------------------------------
// Warp kernel: one sub-group (32 lanes) handles one token.
// TOKENS_PER_WG sub-groups per work-group. Two passes over global memory
// (absmax, then quantize); the pass-2 reload is served from L2 for typical
// hidden sizes. Caching the row in SLM was measured slower — the per-token SLM
// footprint (TOKENS_PER_WG * hidden * 2B) collapses work-group occupancy and
// costs more than the reload it saves. This two-pass global design runs at
// ~85% of the DRAM copy ceiling in the bandwidth-bound (large-batch) regime.
// ---------------------------------------------------------------------------
template <typename T, typename DST_DTYPE, int VEC_SIZE>
class PerTokenQuantFP8WarpKernel {
 private:
  const T* input_;
  DST_DTYPE* output_q_;
  float* output_s_;
  int64_t hidden_dim_;
  int64_t num_tokens_;

 public:
  PerTokenQuantFP8WarpKernel(
      const T* input, DST_DTYPE* output_q, float* output_s, int64_t hidden_dim, int64_t num_tokens)
      : input_(input), output_q_(output_q), output_s_(output_s), hidden_dim_(hidden_dim), num_tokens_(num_tokens) {}

  [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] void operator()(sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    const int warp_id = item.get_local_id(0) / SUB_GROUP_SIZE;
    const int lane_id = item.get_local_id(0) % SUB_GROUP_SIZE;
    const int64_t token_id = static_cast<int64_t>(item.get_group(0)) * TOKENS_PER_WG + warp_id;
    if (token_id >= num_tokens_) return;

    const T* token_input = input_ + token_id * hidden_dim_;
    DST_DTYPE* token_output = output_q_ + token_id * hidden_dim_;
    const int64_t num_vec_elems = hidden_dim_ / VEC_SIZE;

    using vec_in_t = vec_t<T, VEC_SIZE>;
    using output_storage_t = uint8_t;
    using vec_out_t = vec_t<output_storage_t, VEC_SIZE>;

    // Pass 1: strided vectorized load, per-lane absmax.
    float max_value = 0.0f;
    for (int64_t i = lane_id; i < num_vec_elems; i += SUB_GROUP_SIZE) {
      vec_in_t in;
      in.load(0, sycl::multi_ptr<const T, sycl::access::address_space::global_space>(token_input + i * VEC_SIZE));
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        max_value = sycl::fmax(max_value, sycl::fabs(static_cast<float>(in[j])));
      }
    }

    // Reduce absmax across the sub-group (one token).
    max_value = sycl::reduce_over_group(sg, max_value, sycl::maximum<float>());

    const float scale = max_value / FP8_E4M3_MAX;
    if (lane_id == 0) {
      output_s_[token_id] = scale;
    }
    const float scale_inv = (scale == 0.0f) ? 0.0f : 1.0f / scale;

    // Pass 2: reload, quantize, vectorized store.
    for (int64_t i = lane_id; i < num_vec_elems; i += SUB_GROUP_SIZE) {
      vec_in_t in;
      in.load(0, sycl::multi_ptr<const T, sycl::access::address_space::global_space>(token_input + i * VEC_SIZE));
      vec_out_t out;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        float v = sycl::fmax(sycl::fmin(static_cast<float>(in[j]) * scale_inv, FP8_E4M3_MAX), -FP8_E4M3_MAX);
        out[j] = sycl::bit_cast<output_storage_t>(static_cast<DST_DTYPE>(v));
      }
      out.store(
          0,
          sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
              reinterpret_cast<output_storage_t*>(token_output + i * VEC_SIZE)));
    }
  }
};

// ---------------------------------------------------------------------------
// Small-batch kernel: one work-group handles one token, block reduce.
// Used when there are too few tokens to fill the machine with the warp kernel.
// ---------------------------------------------------------------------------
template <typename T, typename DST_DTYPE, int VEC_SIZE>
class PerTokenQuantFP8BlockKernel {
 private:
  const T* input_;
  DST_DTYPE* output_q_;
  float* output_s_;
  int64_t hidden_dim_;
  int64_t num_tokens_;

 public:
  PerTokenQuantFP8BlockKernel(
      const T* input, DST_DTYPE* output_q, float* output_s, int64_t hidden_dim, int64_t num_tokens)
      : input_(input), output_q_(output_q), output_s_(output_s), hidden_dim_(hidden_dim), num_tokens_(num_tokens) {}

  [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] void operator()(sycl::nd_item<1> item) const {
    const int64_t token_id = static_cast<int64_t>(item.get_group(0));
    if (token_id >= num_tokens_) return;

    const int tid = item.get_local_id(0);
    const int block_dim = item.get_local_range(0);

    const T* token_input = input_ + token_id * hidden_dim_;
    DST_DTYPE* token_output = output_q_ + token_id * hidden_dim_;
    const int64_t num_vec_elems = hidden_dim_ / VEC_SIZE;

    using vec_in_t = vec_t<T, VEC_SIZE>;
    using output_storage_t = uint8_t;
    using vec_out_t = vec_t<output_storage_t, VEC_SIZE>;

    float max_value = 0.0f;
    for (int64_t i = tid; i < num_vec_elems; i += block_dim) {
      vec_in_t in;
      in.load(0, sycl::multi_ptr<const T, sycl::access::address_space::global_space>(token_input + i * VEC_SIZE));
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        max_value = sycl::fmax(max_value, sycl::fabs(static_cast<float>(in[j])));
      }
    }

    max_value = sycl::reduce_over_group(item.get_group(), max_value, sycl::maximum<float>());

    const float scale = max_value / FP8_E4M3_MAX;
    if (tid == 0) {
      output_s_[token_id] = scale;
    }
    const float scale_inv = (scale == 0.0f) ? 0.0f : 1.0f / scale;

    for (int64_t i = tid; i < num_vec_elems; i += block_dim) {
      vec_in_t in;
      in.load(0, sycl::multi_ptr<const T, sycl::access::address_space::global_space>(token_input + i * VEC_SIZE));
      vec_out_t out;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        float v = sycl::fmax(sycl::fmin(static_cast<float>(in[j]) * scale_inv, FP8_E4M3_MAX), -FP8_E4M3_MAX);
        out[j] = sycl::bit_cast<output_storage_t>(static_cast<DST_DTYPE>(v));
      }
      out.store(
          0,
          sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
              reinterpret_cast<output_storage_t*>(token_output + i * VEC_SIZE)));
    }
  }
};

void sgl_per_token_quant_fp8(at::Tensor input, at::Tensor output_q, at::Tensor output_s) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);
  TORCH_CHECK(input.dim() == 2, "input must be 2D [num_tokens, hidden_dim]");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::BFloat16 || input.scalar_type() == at::ScalarType::Half,
      "input must be BFloat16/Half tensor");
  TORCH_CHECK(output_q.scalar_type() == at::ScalarType::Float8_e4m3fn, "output_q must be Float8_e4m3fn tensor");
  TORCH_CHECK(output_s.scalar_type() == at::ScalarType::Float, "output_s must be Float tensor");

  const int64_t num_tokens = input.size(0);
  const int64_t hidden_dim = input.size(1);
  TORCH_CHECK(hidden_dim % 4 == 0, "hidden_dim must be divisible by 4, got ", hidden_dim);
  if (num_tokens == 0) return;

  auto& Q = dpcppGetCurrentQueue();
  const int dev_id = dpcppGetDeviceIdOfCurrentQueue();

  // Choose vector width from device preference, capped so hidden_dim stays divisible.
  int vec_size = preferred_vector_width(dev_id, input.element_size());
  while (vec_size > 1 && (hidden_dim % vec_size != 0)) {
    vec_size /= 2;
  }

  // Enough tokens to fill the machine → warp kernel (one sub-group per token).
  // Otherwise the block kernel (one work-group per token) gives more parallelism
  // across the hidden dimension for the few tokens present.
  const int64_t xe_cores = dpcppMaxComputeUnitSize(dev_id);
  const bool use_warp_kernel = num_tokens >= xe_cores * 2 * TOKENS_PER_WG;

#define LAUNCH_WARP(T, DST_DTYPE, VEC)                                        \
  do {                                                                        \
    const int wg_size = TOKENS_PER_WG * SUB_GROUP_SIZE;                       \
    const int64_t num_wgs = (num_tokens + TOKENS_PER_WG - 1) / TOKENS_PER_WG; \
    sycl::range<1> global_range(num_wgs* wg_size);                            \
    sycl::range<1> local_range(wg_size);                                      \
    auto kernel = PerTokenQuantFP8WarpKernel<T, DST_DTYPE, VEC>(              \
        static_cast<const T*>(input.data_ptr()),                              \
        static_cast<DST_DTYPE*>(output_q.data_ptr()),                         \
        output_s.data_ptr<float>(),                                           \
        hidden_dim,                                                           \
        num_tokens);                                                          \
    sycl_kernel_submit(global_range, local_range, Q, kernel);                 \
  } while (0)

#define LAUNCH_BLOCK(T, DST_DTYPE, VEC)                           \
  do {                                                            \
    const int wg_size = 256;                                      \
    sycl::range<1> global_range(num_tokens* wg_size);             \
    sycl::range<1> local_range(wg_size);                          \
    auto kernel = PerTokenQuantFP8BlockKernel<T, DST_DTYPE, VEC>( \
        static_cast<const T*>(input.data_ptr()),                  \
        static_cast<DST_DTYPE*>(output_q.data_ptr()),             \
        output_s.data_ptr<float>(),                               \
        hidden_dim,                                               \
        num_tokens);                                              \
    sycl_kernel_submit(global_range, local_range, Q, kernel);     \
  } while (0)

#define LAUNCH_FOR_VEC(T, DST_DTYPE)      \
  do {                                    \
    switch (vec_size) {                   \
      case 16:                            \
        if (use_warp_kernel)              \
          LAUNCH_WARP(T, DST_DTYPE, 16);  \
        else                              \
          LAUNCH_BLOCK(T, DST_DTYPE, 16); \
        break;                            \
      case 8:                             \
        if (use_warp_kernel)              \
          LAUNCH_WARP(T, DST_DTYPE, 8);   \
        else                              \
          LAUNCH_BLOCK(T, DST_DTYPE, 8);  \
        break;                            \
      case 4:                             \
        if (use_warp_kernel)              \
          LAUNCH_WARP(T, DST_DTYPE, 4);   \
        else                              \
          LAUNCH_BLOCK(T, DST_DTYPE, 4);  \
        break;                            \
      case 2:                             \
        if (use_warp_kernel)              \
          LAUNCH_WARP(T, DST_DTYPE, 2);   \
        else                              \
          LAUNCH_BLOCK(T, DST_DTYPE, 2);  \
        break;                            \
      default:                            \
        if (use_warp_kernel)              \
          LAUNCH_WARP(T, DST_DTYPE, 1);   \
        else                              \
          LAUNCH_BLOCK(T, DST_DTYPE, 1);  \
        break;                            \
    }                                     \
  } while (0)

  AT_DISPATCH_REDUCED_FLOATING_TYPES(input.scalar_type(), "sgl_per_token_quant_fp8", [&] {
    using sycl_scalar_t = typename std::
        conditional<std::is_same<scalar_t, at::Half>::value, sycl::half, sycl::ext::oneapi::bfloat16>::type;
    LAUNCH_FOR_VEC(sycl_scalar_t, cutlass::float_e4m3_t);
  });

#undef LAUNCH_FOR_VEC
#undef LAUNCH_BLOCK
#undef LAUNCH_WARP
}
