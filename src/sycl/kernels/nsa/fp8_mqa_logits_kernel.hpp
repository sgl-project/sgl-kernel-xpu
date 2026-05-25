/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace nsa {

// FP8 e4m3 → float conversion
inline float fp8_e4m3_to_float(uint8_t val) {
  // e4m3: 1 sign, 4 exponent, 3 mantissa, bias=7, no inf, NaN=0x7F/0xFF
  uint32_t sign = (val >> 7) & 1;
  uint32_t exp = (val >> 3) & 0xF;
  uint32_t mant = val & 0x7;

  if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
  // NaN: exp=15, mant=7
  if (exp == 15 && mant == 7) return 0.0f;  // treat NaN as 0

  float result;
  if (exp == 0) {
    // subnormal: value = (-1)^sign * 2^(1-bias) * (0.mant / 8)
    result = static_cast<float>(mant) / 8.0f * (1.0f / 64.0f);  // 2^(1-7) = 2^-6
  } else {
    // normal: value = (-1)^sign * 2^(exp-bias) * (1 + mant/8)
    result = (1.0f + static_cast<float>(mant) / 8.0f);
    int real_exp = static_cast<int>(exp) - 7;
    if (real_exp >= 0)
      result *= static_cast<float>(1 << real_exp);
    else
      result /= static_cast<float>(1 << (-real_exp));
  }
  return sign ? -result : result;
}

// Extract a little-endian float32 from 4 unaligned bytes
inline float load_le_f32(const uint8_t* p) {
  uint32_t bits = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
                  (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
  return *reinterpret_cast<const float*>(&bits);
}

// FP8 MQA logits kernel (prefill/extend path)
// q: (Nq, H, D) fp8 e4m3
// k: (Nk, D) fp8 e4m3
// k_scale: (Nk,) float32
// weights: (Nq, H) float32
// ks: (Nq,) int32 — start index per query
// ke: (Nq,) int32 — end index per query
// out: (Nq, Nk) float32 — must be pre-zeroed for out-of-range positions
//
// For each query i, for each kv position j in [ks[i], ke[i]):
//   score = sum_h( ReLU( dot(q[i,h,:], k[j,:]) ) * weights[i,h] ) * k_scale[j]
struct Fp8MqaLogitsKernel {
  const uint8_t* q_ptr;
  const uint8_t* k_ptr;
  const float* k_scale_ptr;
  const float* weights_ptr;
  const int32_t* ks_ptr;
  const int32_t* ke_ptr;
  float* out_ptr;
  int Nq, Nk, H, D;

  void operator()(sycl::nd_item<2> item) const {
    int qi = item.get_global_id(0);  // query index
    int kj = item.get_global_id(1);  // kv position index

    if (qi >= Nq || kj >= Nk) return;

    int start = ks_ptr[qi];
    int end = ke_ptr[qi];

    if (kj < start || kj >= end) return;  // out-of-range: rely on pre-zeroed output

    float score = 0.0f;

    for (int h = 0; h < H; ++h) {
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) {
        float qv = fp8_e4m3_to_float(q_ptr[(qi * H + h) * D + d]);
        float kv = fp8_e4m3_to_float(k_ptr[kj * D + d]);
        dot += qv * kv;
      }
      // ReLU
      dot = dot > 0.0f ? dot : 0.0f;
      score += dot * weights_ptr[qi * H + h];
    }

    score *= k_scale_ptr[kj];
    out_ptr[qi * Nk + kj] = score;
  }
};

// Paged FP8 MQA logits kernel (decode path)
// q: (B, 1, H, D) fp8 e4m3  — flattened as (B*1*H*D) uint8
// kv_cache: (num_pages, page_size, 1, head_dim_with_sf) uint8
//           head_dim_with_sf = D + 4 (128 fp8 bytes + 4 bytes float32 scale)
// weights: (B, H) float32
// seq_lens: (B,) int32  — actual sequence length per batch
// block_tables: (B, max_num_blocks) int32
// out: (B, max_seq_len) float32 — must be pre-zeroed for out-of-range positions
//
// For each batch b, for each kv position j < seq_lens[b]:
//   page_idx = block_tables[b, j / page_size]
//   token_in_page = j % page_size
//   k_fp8 = kv_cache[page_idx, token_in_page, 0, :D]
//   k_scale = *(float*)&kv_cache[page_idx, token_in_page, 0, D:D+4]
//   score = sum_h( ReLU( dot(q[b,0,h,:], k_fp8) ) * weights[b,h] ) * k_scale
struct Fp8PagedMqaLogitsKernel {
  const uint8_t* q_ptr;             // (B, 1, H, D)
  const uint8_t* kv_cache_ptr;      // (num_pages, page_size, 1, D+4)
  const float* weights_ptr;         // (B, H)
  const int32_t* seq_lens_ptr;      // (B,) or (B,1)
  const int32_t* block_tables_ptr;  // (B, max_num_blocks)
  float* out_ptr;                   // (B, max_seq_len)
  int B, H, D;
  int page_size;
  int max_num_blocks;
  int max_seq_len;

  void operator()(sycl::nd_item<2> item) const {
    int bi = item.get_global_id(0);  // batch index
    int kj = item.get_global_id(1);  // kv position index

    if (bi >= B || kj >= max_seq_len) return;

    int seq_len = seq_lens_ptr[bi];
    if (kj >= seq_len) return;  // out-of-range: rely on pre-zeroed output

    int page_block_idx = kj / page_size;
    int token_in_page = kj % page_size;
    int page_id = block_tables_ptr[bi * max_num_blocks + page_block_idx];

    int head_dim_with_sf = D + 4;
    const uint8_t* kv_token_ptr =
        kv_cache_ptr + page_id * page_size * head_dim_with_sf + token_in_page * head_dim_with_sf;

    float k_scale = load_le_f32(kv_token_ptr + D);

    // q layout: (B, 1, H, D) — the "1" is next_n
    const uint8_t* q_batch = q_ptr + bi * H * D;

    float score = 0.0f;
    for (int h = 0; h < H; ++h) {
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) {
        float qv = fp8_e4m3_to_float(q_batch[h * D + d]);
        float kv = fp8_e4m3_to_float(kv_token_ptr[d]);
        dot += qv * kv;
      }
      dot = dot > 0.0f ? dot : 0.0f;
      score += dot * weights_ptr[bi * H + h];
    }

    score *= k_scale;
    out_ptr[bi * max_seq_len + kj] = score;
  }
};

// Reduction kernel for SYCL-TLA optimized path.
// Takes GEMM output dots(Nq*H, Nk) and reduces across H heads with ReLU + weights.
// dots: (Nq, H, Nk) float32 — reshaped GEMM output
// weights: (Nq, H) float32
// k_scale: (Nk,) float32
// ks: (Nq,) int32
// ke: (Nq,) int32
// out: (Nq, Nk) float32 — must be pre-zeroed for out-of-range positions
struct Fp8MqaLogitsReduceKernel {
  const float* dots_ptr;
  const float* weights_ptr;
  const float* k_scale_ptr;
  const int32_t* ks_ptr;
  const int32_t* ke_ptr;
  float* out_ptr;
  int Nq, H, Nk;

  void operator()(sycl::nd_item<2> item) const {
    int qi = item.get_global_id(0);
    int kj = item.get_global_id(1);
    if (qi >= Nq || kj >= Nk) return;

    if (kj < ks_ptr[qi] || kj >= ke_ptr[qi]) return;  // rely on pre-zeroed output

    float score = 0.0f;
    for (int h = 0; h < H; ++h) {
      float dot = dots_ptr[(qi * H + h) * Nk + kj];
      dot = dot > 0.0f ? dot : 0.0f;  // ReLU
      score += dot * weights_ptr[qi * H + h];
    }
    score *= k_scale_ptr[kj];
    out_ptr[qi * Nk + kj] = score;
  }
};

// Gather kernel: extracts FP8 keys and scales from paged KV cache
// into contiguous buffers for GEMM.
// kv_cache: (num_pages, page_size, 1, D+4) uint8
// block_tables: (B, max_num_blocks) int32
// seq_lens: (B,) int32
// k_out: (B * max_seq_len, D) uint8 — contiguous FP8 keys
// k_scale_out: (B, max_seq_len) float32
struct PagedKGatherKernel {
  const uint8_t* kv_cache_ptr;
  const int32_t* block_tables_ptr;
  const int32_t* seq_lens_ptr;
  uint8_t* k_out_ptr;
  float* k_scale_out_ptr;
  int B, D, page_size, max_num_blocks, max_seq_len;
  int head_dim_with_sf;

  void operator()(sycl::nd_item<2> item) const {
    int b = item.get_global_id(0);
    int kj = item.get_global_id(1);
    if (b >= B || kj >= max_seq_len) return;

    int out_idx = b * max_seq_len + kj;
    if (kj >= seq_lens_ptr[b]) {
      // Zero padding for out-of-range tokens
      auto* dst = reinterpret_cast<uint32_t*>(k_out_ptr + out_idx * D);
      for (int i = 0; i < D / 4; ++i)
        dst[i] = 0;
      k_scale_out_ptr[out_idx] = 0.0f;
      return;
    }

    int page_block = kj / page_size;
    int token_in_page = kj % page_size;
    int page_id = block_tables_ptr[b * max_num_blocks + page_block];

    const uint8_t* src = kv_cache_ptr + page_id * page_size * head_dim_with_sf + token_in_page * head_dim_with_sf;

    // Vectorized copy: 4 bytes at a time (D is always a multiple of 4)
    auto* dst = reinterpret_cast<uint32_t*>(k_out_ptr + out_idx * D);
    auto* src32 = reinterpret_cast<const uint32_t*>(src);
    for (int i = 0; i < D / 4; ++i)
      dst[i] = src32[i];

    k_scale_out_ptr[out_idx] = load_le_f32(src + D);
  }
};

// Reduction kernel for paged path (after batched GEMM).
// dots: (B, H, max_seq_len) float32
// weights: (B, H) float32
// k_scale: (B, max_seq_len) float32
// seq_lens: (B,) int32
// out: (B, max_seq_len) float32 — must be pre-zeroed for out-of-range positions
struct Fp8PagedMqaLogitsReduceKernel {
  const float* dots_ptr;
  const float* weights_ptr;
  const float* k_scale_ptr;
  const int32_t* seq_lens_ptr;
  float* out_ptr;
  int B, H, max_seq_len;

  void operator()(sycl::nd_item<2> item) const {
    int bi = item.get_global_id(0);
    int kj = item.get_global_id(1);
    if (bi >= B || kj >= max_seq_len) return;

    if (kj >= seq_lens_ptr[bi]) return;  // rely on pre-zeroed output

    float score = 0.0f;
    for (int h = 0; h < H; ++h) {
      float dot = dots_ptr[(bi * H + h) * max_seq_len + kj];
      dot = dot > 0.0f ? dot : 0.0f;
      score += dot * weights_ptr[bi * H + h];
    }
    score *= k_scale_ptr[bi * max_seq_len + kj];
    out_ptr[bi * max_seq_len + kj] = score;
  }
};

}  // namespace nsa
