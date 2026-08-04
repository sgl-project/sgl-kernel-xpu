#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "Compress.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace at::native::xpu {

namespace {

constexpr int64_t kHeadDimIndexer = 128;
constexpr int64_t kHeadDimFlashMLA = 512;
constexpr int64_t kRopeDim = 64;
constexpr float kFp8E4m3Max = 448.0f;
constexpr int64_t kNopeDimFlashMLA = kHeadDimFlashMLA - kRopeDim;  // 448
constexpr int64_t kNopeWarpsFlashMLA = 7;
constexpr int64_t kElemsPerNopeWarp = kNopeDimFlashMLA / kNopeWarpsFlashMLA;  // 64

constexpr uint32_t kIndexerLocalSize = 64;
constexpr uint32_t kFlashMLALocalSize = 256;

inline int64_t flashmla_page_bytes(int64_t page_size) {
  return ((584 * page_size + 575) / 576) * 576;
}

inline bool is_power_of_two(int64_t v) {
  return v > 0 && (v & (v - 1)) == 0;
}

inline int64_t log2_i64(int64_t v) {
  int64_t bits = 0;
  while ((1LL << bits) < v) {
    ++bits;
  }
  return bits;
}

template <typename T>
inline float to_float(T v) {
  return static_cast<float>(v);
}

inline uint8_t cast_to_ue8m0(float x) {
  const float clamped = sycl::fmax(x, 1.0e-38f);
  const uint32_t bits = sycl::bit_cast<uint32_t>(clamped);
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xFF);
  const uint32_t mant = bits & 0x7FFFFF;
  exp += (mant != 0u) ? 1 : 0;
  if (exp > 255) {
    exp = 255;
  }
  return static_cast<uint8_t>(exp);
}

inline float inv_scale_ue8m0(uint8_t ue8m0) {
  const int32_t inv_exp = sycl::max(0, 254 - static_cast<int32_t>(ue8m0));
  const uint32_t inv_bits = static_cast<uint32_t>(inv_exp) << 23;
  if (ue8m0 >= 254) {
    return 0.0f;
  }
  return sycl::bit_cast<float>(inv_bits);
}

inline uint8_t quant_fp4_e2m1(float x) {
  const float ax = sycl::fmin(sycl::fabs(x), 6.0f);
  uint8_t idx = 0;
  idx += ax > 0.25f;
  idx += ax > 0.75f;
  idx += ax > 1.25f;
  idx += ax > 1.75f;
  idx += ax > 2.5f;
  idx += ax > 3.5f;
  idx += ax > 5.0f;
  if (x < 0.0f && idx != 0) {
    idx |= 0x8;
  }
  return idx;
}

inline uint8_t cvt_float_to_fp8_e4m3(float x) {
  x = sycl::fmax(sycl::fmin(x, kFp8E4m3Max), -kFp8E4m3Max);
  if (x == 0.0f) {
    return 0;
  }

  const uint32_t u = sycl::bit_cast<uint32_t>(x);
  const uint8_t sign = static_cast<uint8_t>((u >> 31) & 1u);
  const int32_t exp32 = static_cast<int32_t>((u >> 23) & 0xFFu) - 127;
  const int32_t mant23 = static_cast<int32_t>(u & 0x7FFFFFu);

  constexpr int32_t bias = 7;
  constexpr int32_t max_exp = 15;
  constexpr int32_t min_sub = -9;
  constexpr int32_t min_norm = -6;
  constexpr uint8_t saturate = 0x7E;

  if (exp32 < min_sub) {
    return static_cast<uint8_t>(sign << 7);
  }

  int32_t exp8 = 0;
  int32_t mant3 = 0;

  if (exp32 < min_norm) {
    const int32_t shift = (-(bias - 1) - exp32);
    const int32_t base = 0x800000 | mant23;
    int32_t subnorm_mant = base >> (shift + 20);
    const int32_t round_bit = (base >> (shift + 19)) & 1;
    subnorm_mant += round_bit;

    if (subnorm_mant > 7) {
      exp8 = 1;
      mant3 = 0;
    } else {
      exp8 = 0;
      mant3 = subnorm_mant & 0x7;
    }
  } else {
    exp8 = exp32 + bias;
    mant3 = (mant23 >> 20) + ((mant23 >> 19) & 1);
    if (mant3 > 7) {
      mant3 = 0;
      exp8 += 1;
    }
    if (exp8 > max_exp) {
      exp8 = static_cast<int32_t>(saturate >> 3);
      mant3 = static_cast<int32_t>(saturate & 0x7);
    }
  }

  return static_cast<uint8_t>((sign << 7) | ((exp8 & 0x1F) << 3) | (mant3 & 0x7));
}

inline uint16_t float_to_bf16_bits(float x) {
  sycl::ext::oneapi::bfloat16 bf = static_cast<sycl::ext::oneapi::bfloat16>(x);
  return sycl::bit_cast<uint16_t>(bf);
}

inline uint16_t pack_u8x2(uint8_t lo, uint8_t hi) {
  return static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8);
}

inline float subgroup_xor_reduce_sum_16(sycl::sub_group sg, float v) {
  v += sycl::permute_group_by_xor(sg, v, 8);
  v += sycl::permute_group_by_xor(sg, v, 4);
  v += sycl::permute_group_by_xor(sg, v, 2);
  v += sycl::permute_group_by_xor(sg, v, 1);
  return v;
}

inline float subgroup_xor_reduce_max_16(sycl::sub_group sg, float v) {
  v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, 8));
  v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, 4));
  v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, 2));
  v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, 1));
  return v;
}

template <uint32_t kLocalSize>
inline void local_reduce_sum_pow2(float* red, sycl::nd_item<1> item, uint32_t tid) {
  for (uint32_t stride = kLocalSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      red[tid] += red[tid + stride];
    }
    item.barrier(sycl::access::fence_space::local_space);
  }
}

template <uint32_t kLocalSize>
inline void local_reduce_max_pow2(float* red, sycl::nd_item<1> item, uint32_t tid) {
  for (uint32_t stride = kLocalSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      red[tid] = sycl::fmax(red[tid], red[tid + stride]);
    }
    item.barrier(sycl::access::fence_space::local_space);
  }
}

inline void local_reduce_sum_64_sg16(float* red, sycl::nd_item<1> item, sycl::sub_group sg, uint32_t tid) {
  const uint32_t lane = tid & 0xFu;
  const uint32_t sg_id = tid >> 4;
  const float sg_sum = subgroup_xor_reduce_sum_16(sg, red[tid]);
  if (lane == 0) {
    red[sg_id] = sg_sum;
  }
  item.barrier(sycl::access::fence_space::local_space);

  if (sg_id == 0) {
    const float block_sum = subgroup_xor_reduce_sum_16(sg, (lane < 4u) ? red[lane] : 0.0f);
    if (lane == 0) {
      red[0] = block_sum;
    }
  }
  item.barrier(sycl::access::fence_space::local_space);
}

inline void local_reduce_max_64_sg16(float* red, sycl::nd_item<1> item, sycl::sub_group sg, uint32_t tid) {
  const uint32_t lane = tid & 0xFu;
  const uint32_t sg_id = tid >> 4;
  const float sg_max = subgroup_xor_reduce_max_16(sg, red[tid]);
  if (lane == 0) {
    red[sg_id] = sg_max;
  }
  item.barrier(sycl::access::fence_space::local_space);

  if (sg_id == 0) {
    const float block_max = subgroup_xor_reduce_max_16(sg, (lane < 4u) ? red[lane] : 0.0f);
    if (lane == 0) {
      red[0] = block_max;
    }
  }
  item.barrier(sycl::access::fence_space::local_space);
}

template <typename input_t>
struct FusedNormRopeIndexerKernel {
  // Indexer variant: head_dim=128, one token per work-group (64 threads).
  // Each lane handles 2 elements, covering all 128 dims per token.
  // Cache layout per token:
  //  - FP8 path: 128 value bytes + 4-byte fp32 scale
  //  - FP4 path: 64 packed bytes + 4 UE8M0 scale bytes
  [[sycl::reqd_sub_group_size(16)]]
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t group_id = static_cast<uint32_t>(item.get_group(0));
    const uint32_t tid = static_cast<uint32_t>(item.get_local_id(0));
    if (group_id >= num_tokens) {
      return;
    }

    int64_t position = 0;
    int64_t slot = -1;
    bool active = false;

    if (is_decode) {
      const DecodePlan plan = plan_d[group_id];
      if (plan.seq_len != 0 && plan.seq_len % compress_ratio == 0) {
        position = static_cast<int64_t>(plan.seq_len) - static_cast<int64_t>(compress_ratio);
        slot = out_loc[group_id];
        active = true;
      }
    } else {
      const CompressPlan plan = plan_c[group_id];
      if (!plan.is_invalid()) {
        position = static_cast<int64_t>(plan.seq_len) - static_cast<int64_t>(compress_ratio);
        slot = out_loc[static_cast<int64_t>(plan.ragged_id)];
        active = true;
      }
    }

    if (!active || slot < 0) {
      return;
    }

    float* data = smem_data.template get_multi_ptr<sycl::access::decorated::no>().get();
    float* red = smem_red.template get_multi_ptr<sycl::access::decorated::no>().get();
    float* group_inv = smem_group_inv.template get_multi_ptr<sycl::access::decorated::no>().get();
    uint8_t* group_ue8 = smem_group_ue8.template get_multi_ptr<sycl::access::decorated::no>().get();
    auto sg = item.get_sub_group();

    const input_t* row_in = input + static_cast<int64_t>(group_id) * kHeadDimIndexer;

    const int64_t d0 = static_cast<int64_t>(tid);
    const int64_t d1 = d0 + static_cast<int64_t>(kIndexerLocalSize);

    // Part 1: RMSNorm over the full 128-d vector.
    const float x0 = to_float(row_in[d0]);
    const float x1 = to_float(row_in[d1]);
    data[d0] = x0;
    data[d1] = x1;
    const float local_sum = x0 * x0 + x1 * x1;

    red[tid] = local_sum;
    local_reduce_sum_64_sg16(red, item, sg, tid);

    const float norm_factor = sycl::rsqrt(red[0] / static_cast<float>(kHeadDimIndexer) + eps);
    data[d0] = data[d0] * norm_factor * to_float(weight[d0]);
    data[d1] = data[d1] * norm_factor * to_float(weight[d1]);
    item.barrier(sycl::access::fence_space::local_space);

    // Part 2: RoPE on the tail 64 dims.
    const float* freq = freqs_cis + position * kRopeDim;
    if (tid < static_cast<uint32_t>(kRopeDim / 2)) {
      const int64_t p = static_cast<int64_t>(tid);
      const int64_t base = 64 + 2 * p;
      const float xr = data[base + 0];
      const float xi = data[base + 1];
      const float fr = freq[2 * p + 0];
      const float fi = freq[2 * p + 1];
      data[base + 0] = xr * fr - xi * fi;
      data[base + 1] = xr * fi + xi * fr;
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Part 3: Hadamard-128.
    // 1) register-local pair butterfly (a+b, a-b)
    // 2) intra-subgroup XOR butterflies (mask 1/2/4/8)
    // 3) cross-subgroup butterflies (mask 16/32) via local memory
    float h0 = data[d0] + data[d1];
    float h1 = data[d0] - data[d1];

    const uint32_t lane = tid & 0xFu;
    for (uint32_t mask = 1; mask <= 8; mask <<= 1) {
      const float o0 = sycl::permute_group_by_xor(sg, h0, mask);
      const float o1 = sycl::permute_group_by_xor(sg, h1, mask);
      h0 = (lane & mask) ? (o0 - h0) : (h0 + o0);
      h1 = (lane & mask) ? (o1 - h1) : (h1 + o1);
    }

    data[d0] = h0;
    data[d1] = h1;
    item.barrier(sycl::access::fence_space::local_space);

    for (uint32_t mask = 16; mask <= 32; mask <<= 1) {
      const uint32_t peer = tid ^ mask;
      const float o0 = data[static_cast<int64_t>(peer)];
      const float o1 = data[static_cast<int64_t>(peer) + kIndexerLocalSize];
      h0 = (tid & mask) ? (o0 - h0) : (h0 + o0);
      h1 = (tid & mask) ? (o1 - h1) : (h1 + o1);
      data[d0] = h0;
      data[d1] = h1;
      item.barrier(sycl::access::fence_space::local_space);
    }

    // 1 / sqrt(128)
    constexpr float kHadamardScale = 0.08838834764831845f;
    data[d0] = h0 * kHadamardScale;
    data[d1] = h1 * kHadamardScale;
    item.barrier(sycl::access::fence_space::local_space);

    const int64_t page = slot >> page_bits;
    const int64_t offset = slot & (page_size - 1);

    // Part 4a: FP8 store. For preshuffle_size>0, values are written in tiled
    // order to match the pre-shuffled consumer layout.
    if (!use_fp4) {
      float local_max = sycl::fmax(sycl::fabs(data[d0]), sycl::fabs(data[d1]));
      red[tid] = local_max;
      local_reduce_max_64_sg16(red, item, sg, tid);

      const float scale = sycl::fmax(1.0e-4f, red[0]) / kFp8E4m3Max;
      const float inv_scale = 1.0f / scale;
      const int64_t scale_base = page * page_bytes + 128 * page_size + offset * 4;
      for (int64_t pair = tid; pair < kHeadDimIndexer / 2; pair += kIndexerLocalSize) {
        const int64_t i0 = 2 * pair;
        const int64_t i1 = i0 + 1;
        const uint8_t q0 = cvt_float_to_fp8_e4m3(data[i0] * inv_scale);
        const uint8_t q1 = cvt_float_to_fp8_e4m3(data[i1] * inv_scale);

        if (preshuffle_size == 0) {
          const int64_t value_base = page * page_bytes + offset * 128;
          reinterpret_cast<uint16_t*>(kvcache + value_base)[pair] = pack_u8x2(q0, q1);
          continue;
        }

        const int64_t token_tile_id = offset / preshuffle_size;
        const int64_t token_in_tile = offset % preshuffle_size;

        const int64_t col_tile_id0 = i0 / preshuffle_size;
        const int64_t col_in_tile0 = i0 % preshuffle_size;
        const int64_t value_offset0 = token_tile_id * (preshuffle_size * kHeadDimIndexer) +
                                      col_tile_id0 * (preshuffle_size * preshuffle_size) +
                                      token_in_tile * preshuffle_size + col_in_tile0;
        kvcache[page * page_bytes + value_offset0] = q0;

        const int64_t col_tile_id1 = i1 / preshuffle_size;
        const int64_t col_in_tile1 = i1 % preshuffle_size;
        const int64_t value_offset1 = token_tile_id * (preshuffle_size * kHeadDimIndexer) +
                                      col_tile_id1 * (preshuffle_size * preshuffle_size) +
                                      token_in_tile * preshuffle_size + col_in_tile1;
        kvcache[page * page_bytes + value_offset1] = q1;
      }

      if (tid == 0) {
        reinterpret_cast<uint32_t*>(kvcache + scale_base)[0] = sycl::bit_cast<uint32_t>(scale);
      }
      return;
    }

    // Part 4b: FP4 store (4 groups x 32 dims), each group owns one UE8M0 scale.
    for (int64_t g = 0; g < 4; ++g) {
      float g_local_max = 0.0f;
      if (tid < 32u) {
        g_local_max = sycl::fabs(data[g * 32 + static_cast<int64_t>(tid)]);
      }
      red[tid] = g_local_max;
      local_reduce_max_64_sg16(red, item, sg, tid);

      if (tid == 0) {
        const float scale_raw = sycl::fmax(1.0e-4f, red[0]) / 6.0f;
        const uint8_t ue8 = cast_to_ue8m0(scale_raw);
        group_ue8[g] = ue8;
        group_inv[g] = inv_scale_ue8m0(ue8);
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    const int64_t value_base = page * page_bytes + offset * 64;
    const int64_t scale_base = page * page_bytes + 64 * page_size + offset * 4;

    const int64_t i = static_cast<int64_t>(tid);
    const int64_t i0 = 2 * i;
    const int64_t i1 = i0 + 1;
    const int64_t g0 = i0 / 32;
    const int64_t g1 = i1 / 32;
    const uint8_t q0 = quant_fp4_e2m1(data[i0] * group_inv[g0]);
    const uint8_t q1 = quant_fp4_e2m1(data[i1] * group_inv[g1]);
    kvcache[value_base + i] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));

    if (tid < 4) {
      kvcache[scale_base + tid] = group_ue8[tid];
    }
  }

  const input_t* input;
  const input_t* weight;
  const float* freqs_cis;
  const int64_t* out_loc;
  uint8_t* kvcache;
  const DecodePlan* plan_d;
  const CompressPlan* plan_c;
  uint32_t num_tokens;
  uint32_t compress_ratio;
  int64_t page_size;
  int64_t page_bits;
  int64_t page_bytes;
  float eps;
  bool is_decode;
  bool use_fp4;
  int64_t preshuffle_size;
  sycl::local_accessor<float, 1> smem_data;
  sycl::local_accessor<float, 1> smem_red;
  sycl::local_accessor<float, 1> smem_group_inv;
  sycl::local_accessor<uint8_t, 1> smem_group_ue8;
};

template <typename input_t>
struct FusedNormRopeFlashMLAKernel {
  // FlashMLA variant: head_dim=512, one token per work-group (256 threads).
  // Each lane handles 2 elements, covering all 512 dims per token.
  // Default cache layout per token: 576 value bytes (448 FP8 NoPE + 64 BF16 RoPE)
  // plus 8 scale bytes (7 used, 1 padding).
  [[sycl::reqd_sub_group_size(16)]]
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t group_id = static_cast<uint32_t>(item.get_group(0));
    const uint32_t tid = static_cast<uint32_t>(item.get_local_id(0));
    if (group_id >= num_tokens) {
      return;
    }

    int64_t position = 0;
    int64_t slot = -1;
    bool active = false;

    if (is_decode) {
      const DecodePlan plan = plan_d[group_id];
      if (plan.seq_len != 0 && plan.seq_len % compress_ratio == 0) {
        position = static_cast<int64_t>(plan.seq_len) - static_cast<int64_t>(compress_ratio);
        slot = out_loc[group_id];
        active = true;
      }
    } else {
      const CompressPlan plan = plan_c[group_id];
      if (!plan.is_invalid()) {
        position = static_cast<int64_t>(plan.seq_len) - static_cast<int64_t>(compress_ratio);
        slot = out_loc[static_cast<int64_t>(plan.ragged_id)];
        active = true;
      }
    }

    if (!active || slot < 0) {
      return;
    }

    float* data = smem_data.template get_multi_ptr<sycl::access::decorated::no>().get();
    float* red = smem_red.template get_multi_ptr<sycl::access::decorated::no>().get();
    float* group_inv = smem_group_inv.template get_multi_ptr<sycl::access::decorated::no>().get();
    uint8_t* group_ue8 = smem_group_ue8.template get_multi_ptr<sycl::access::decorated::no>().get();
    auto sg = item.get_sub_group();

    const input_t* row_in = input + static_cast<int64_t>(group_id) * kHeadDimFlashMLA;

    // Part 1: RMSNorm over head_dim=512 using subgroup partial reductions.
    float local_sum = 0.0f;
    const int64_t d0 = static_cast<int64_t>(tid) * 2;
    const int64_t d1 = d0 + 1;
    if (d0 < kHeadDimFlashMLA) {
      const float x0 = to_float(row_in[d0]);
      data[d0] = x0;
      local_sum += x0 * x0;
    }
    if (d1 < kHeadDimFlashMLA) {
      const float x1 = to_float(row_in[d1]);
      data[d1] = x1;
      local_sum += x1 * x1;
    }

    const uint32_t lane = tid & 0xFu;
    const uint32_t sg_id = tid >> 4;
    float sg_sum = subgroup_xor_reduce_sum_16(sg, local_sum);
    if (lane == 0) {
      red[sg_id] = sg_sum;
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (sg_id == 0) {
      float block_sum = (lane < static_cast<uint32_t>(kFlashMLALocalSize / 16)) ? red[lane] : 0.0f;
      block_sum = subgroup_xor_reduce_sum_16(sg, block_sum);
      if (lane == 0) {
        red[0] = block_sum;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    const float norm_factor = sycl::rsqrt(red[0] / static_cast<float>(kHeadDimFlashMLA) + eps);
    if (d0 < kHeadDimFlashMLA) {
      data[d0] = data[d0] * norm_factor * to_float(weight[d0]);
    }
    if (d1 < kHeadDimFlashMLA) {
      data[d1] = data[d1] * norm_factor * to_float(weight[d1]);
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Part 2: RoPE on the tail 64 dims.
    const float* freq = freqs_cis + position * kRopeDim;
    for (int64_t p = tid; p < kRopeDim / 2; p += kFlashMLALocalSize) {
      const int64_t base = kNopeDimFlashMLA + 2 * p;
      const float xr = data[base + 0];
      const float xi = data[base + 1];
      const float fr = freq[2 * p + 0];
      const float fi = freq[2 * p + 1];
      data[base + 0] = xr * fr - xi * fi;
      data[base + 1] = xr * fi + xi * fr;
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (use_bf16_store) {
      // Optional mode: write the whole 512-d output as plain BF16.
      const int64_t page = slot >> page_bits;
      const int64_t offset = slot & (page_size - 1);
      const int64_t value_base = page * page_bytes + offset * (kHeadDimFlashMLA * 2);
      uint16_t* value_ptr = reinterpret_cast<uint16_t*>(kvcache + value_base);
      if (d0 < kHeadDimFlashMLA) {
        value_ptr[d0] = float_to_bf16_bits(data[d0]);
      }
      if (d1 < kHeadDimFlashMLA) {
        value_ptr[d1] = float_to_bf16_bits(data[d1]);
      }
      return;
    }

    const int64_t g_d0 = (d0 < kNopeDimFlashMLA) ? (d0 / kElemsPerNopeWarp) : -1;
    const int64_t g_d1 = (d1 < kNopeDimFlashMLA) ? (d1 / kElemsPerNopeWarp) : -1;

    // Part 3: NoPE FP8 quantization. Each 64-d group gets one UE8M0 scale.
#pragma unroll
    for (int64_t g = 0; g < kNopeWarpsFlashMLA; ++g) {
      float g_local_max = 0.0f;
      if (g_d0 == g) {
        g_local_max = sycl::fabs(data[d0]);
      }
      if (g_d1 == g) {
        g_local_max = sycl::fmax(g_local_max, sycl::fabs(data[d1]));
      }

      const float sg_max = subgroup_xor_reduce_max_16(sg, g_local_max);
      if (lane == 0) {
        red[sg_id] = sg_max;
      }
      item.barrier(sycl::access::fence_space::local_space);

      if (tid == 0) {
        const uint32_t sg0 = static_cast<uint32_t>(g * 2);
        const float abs_max = sycl::fmax(red[sg0], red[sg0 + 1]);
        const float scale_raw = sycl::fmax(1.0e-4f, abs_max) / kFp8E4m3Max;
        const uint8_t ue8 = cast_to_ue8m0(scale_raw);
        group_ue8[g] = ue8;
        group_inv[g] = inv_scale_ue8m0(ue8);
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    const int64_t page = slot >> page_bits;
    const int64_t offset = slot & (page_size - 1);
    const int64_t value_base = page * page_bytes + offset * 576;
    const int64_t scale_base = page * page_bytes + 576 * page_size + offset * 8;

    // NoPE values: 448 FP8 bytes packed as uint16 pairs.
    uint16_t* nope_ptr = reinterpret_cast<uint16_t*>(kvcache + value_base);
    if (tid < static_cast<uint32_t>(kNopeDimFlashMLA / 2)) {
      const int64_t pair = static_cast<int64_t>(tid);
      const int64_t i0 = 2 * pair;
      const int64_t i1 = i0 + 1;
      const int64_t g0 = i0 / kElemsPerNopeWarp;
      const int64_t g1 = i1 / kElemsPerNopeWarp;
      const uint8_t q0 = cvt_float_to_fp8_e4m3(data[i0] * group_inv[g0]);
      const uint8_t q1 = cvt_float_to_fp8_e4m3(data[i1] * group_inv[g1]);
      nope_ptr[pair] = pack_u8x2(q0, q1);
    }

    // RoPE tail: 64 BF16 values.
    uint16_t* rope_ptr = reinterpret_cast<uint16_t*>(kvcache + value_base + kNopeDimFlashMLA);
    if (tid < static_cast<uint32_t>(kRopeDim)) {
      const int64_t i = static_cast<int64_t>(tid);
      rope_ptr[i] = float_to_bf16_bits(data[kNopeDimFlashMLA + i]);
    }

    // Scale region: first 7 bytes are valid scales (one per NoPE group).
    if (tid < static_cast<uint32_t>(kNopeWarpsFlashMLA)) {
      kvcache[scale_base + tid] = group_ue8[tid];
    }
  }

  const input_t* input;
  const input_t* weight;
  const float* freqs_cis;
  const int64_t* out_loc;
  uint8_t* kvcache;
  const DecodePlan* plan_d;
  const CompressPlan* plan_c;
  uint32_t num_tokens;
  uint32_t compress_ratio;
  int64_t page_size;
  int64_t page_bits;
  int64_t page_bytes;
  float eps;
  bool is_decode;
  bool use_bf16_store;
  sycl::local_accessor<float, 1> smem_data;
  sycl::local_accessor<float, 1> smem_red;
  sycl::local_accessor<float, 1> smem_group_inv;
  sycl::local_accessor<uint8_t, 1> smem_group_ue8;
};

}  // namespace

SGL_KERNEL_EXPORT void fused_norm_rope_store(
    torch::Tensor input,
    torch::Tensor plan,
    torch::Tensor norm_weight,
    double norm_eps,
    torch::Tensor freq_cis,
    torch::Tensor out_loc,
    torch::Tensor kvcache,
    bool is_decode,
    int64_t compress_ratio,
    int64_t page_size,
    bool use_fp4,
    int64_t preshuffle_size,
    bool use_bf16_store) {
  TORCH_CHECK(
      input.is_xpu() && input.dim() == 2 && input.is_contiguous(), "input must be contiguous [N, head_dim] XPU tensor");
  TORCH_CHECK(
      plan.is_xpu() && plan.dtype() == torch::kUInt8 && plan.dim() == 2 && plan.is_contiguous(),
      "plan must be contiguous [N, 16] uint8 XPU tensor");
  TORCH_CHECK(
      norm_weight.is_xpu() && norm_weight.dim() == 1 && norm_weight.is_contiguous(),
      "norm_weight must be contiguous [head_dim] XPU tensor");
  TORCH_CHECK(
      freq_cis.is_xpu() && freq_cis.dtype() == torch::kFloat && freq_cis.dim() == 2 && freq_cis.is_contiguous(),
      "freq_cis must be contiguous [max_pos, 64] float32 XPU tensor");
  TORCH_CHECK(
      out_loc.is_xpu() && out_loc.dim() == 1 && out_loc.is_contiguous(), "out_loc must be contiguous [M] XPU tensor");
  TORCH_CHECK(
      kvcache.is_xpu() && kvcache.dtype() == torch::kUInt8 && kvcache.dim() == 2 && kvcache.is_contiguous(),
      "kvcache must be contiguous [num_pages, page_bytes] uint8 XPU tensor");

  const int64_t num_tokens = input.size(0);
  const int64_t head_dim = input.size(1);

  TORCH_CHECK(
      head_dim == kHeadDimIndexer || head_dim == kHeadDimFlashMLA, "head_dim must be 128 or 512, got ", head_dim);
  TORCH_CHECK(norm_weight.size(0) == head_dim, "norm_weight size must equal head_dim");
  TORCH_CHECK(freq_cis.size(1) == kRopeDim, "freq_cis last dim must be 64");
  TORCH_CHECK(
      plan.size(0) == num_tokens && plan.size(1) == static_cast<int64_t>(sizeof(DecodePlan)), "plan must be [N, 16]");
  TORCH_CHECK(compress_ratio > 0, "compress_ratio must be > 0");
  TORCH_CHECK(is_power_of_two(page_size), "page_size must be power of 2");

  TORCH_CHECK(input.scalar_type() == norm_weight.scalar_type(), "input and norm_weight dtypes must match");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16 ||
          input.scalar_type() == at::ScalarType::Float,
      "input dtype must be fp16/bf16/fp32");

  if (use_fp4) {
    TORCH_CHECK(head_dim == kHeadDimIndexer, "use_fp4 is only supported for head_dim=128");
  }

  TORCH_CHECK(preshuffle_size >= 0, "preshuffle_size must be >= 0");
  if (preshuffle_size > 0) {
    TORCH_CHECK(head_dim == kHeadDimIndexer, "preshuffle_size is only supported for head_dim=128");
    TORCH_CHECK(!use_fp4, "preshuffle_size is not supported with use_fp4=True");
    TORCH_CHECK(preshuffle_size % 2 == 0, "preshuffle_size must be even");
    TORCH_CHECK(kHeadDimIndexer % preshuffle_size == 0, "head_dim(128) must be divisible by preshuffle_size");
    TORCH_CHECK(page_size % preshuffle_size == 0, "page_size must be divisible by preshuffle_size");
  }

  if (use_bf16_store) {
    TORCH_CHECK(head_dim == kHeadDimFlashMLA, "use_bf16_store is only supported for head_dim=512");
    TORCH_CHECK(!use_fp4, "use_bf16_store is not supported with use_fp4=True");
  }

  const int64_t expected_page_bytes =
      (head_dim == kHeadDimIndexer)
          ? ((use_fp4 ? 68 : 132) * page_size)
          : (use_bf16_store ? (kHeadDimFlashMLA * 2 * page_size) : flashmla_page_bytes(page_size));
  TORCH_CHECK(
      kvcache.size(1) == expected_page_bytes,
      "kvcache page_bytes mismatch. expected ",
      expected_page_bytes,
      ", got ",
      kvcache.size(1));

  if (num_tokens == 0) {
    return;
  }

  auto out_loc_i64 = out_loc.scalar_type() == at::ScalarType::Long ? out_loc : out_loc.to(torch::kLong);

  const auto* plan_d = reinterpret_cast<const DecodePlan*>(plan.data_ptr<uint8_t>());
  const auto* plan_c = reinterpret_cast<const CompressPlan*>(plan.data_ptr<uint8_t>());

  const int64_t page_bits = log2_i64(page_size);
  auto queue = c10::xpu::getCurrentXPUStream().queue();

  SYCL_DISPATCH_FLOATING_TYPES(at::kHalf, at::kBFloat16, input.scalar_type(), "fused_norm_rope_store", [&]() {
    using input_t = scalar_t;

    if (head_dim == kHeadDimIndexer) {
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> smem_data(sycl::range<1>(kHeadDimIndexer), cgh);
        sycl::local_accessor<float, 1> smem_red(sycl::range<1>(kIndexerLocalSize), cgh);
        sycl::local_accessor<float, 1> smem_group_inv(sycl::range<1>(4), cgh);
        sycl::local_accessor<uint8_t, 1> smem_group_ue8(sycl::range<1>(4), cgh);

        FusedNormRopeIndexerKernel<input_t> kernel{
            input.data_ptr<input_t>(),
            norm_weight.data_ptr<input_t>(),
            freq_cis.data_ptr<float>(),
            out_loc_i64.data_ptr<int64_t>(),
            kvcache.data_ptr<uint8_t>(),
            plan_d,
            plan_c,
            static_cast<uint32_t>(num_tokens),
            static_cast<uint32_t>(compress_ratio),
            page_size,
            page_bits,
            kvcache.stride(0),
            static_cast<float>(norm_eps),
            is_decode,
            use_fp4,
            preshuffle_size,
            smem_data,
            smem_red,
            smem_group_inv,
            smem_group_ue8,
        };

        const uint32_t global_size = static_cast<uint32_t>(num_tokens) * kIndexerLocalSize;
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kIndexerLocalSize)), kernel);
      });
      return;
    }

    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> smem_data(sycl::range<1>(kHeadDimFlashMLA), cgh);
      sycl::local_accessor<float, 1> smem_red(sycl::range<1>(kFlashMLALocalSize), cgh);
      sycl::local_accessor<float, 1> smem_group_inv(sycl::range<1>(kNopeWarpsFlashMLA), cgh);
      sycl::local_accessor<uint8_t, 1> smem_group_ue8(sycl::range<1>(kNopeWarpsFlashMLA), cgh);

      FusedNormRopeFlashMLAKernel<input_t> kernel{
          input.data_ptr<input_t>(),
          norm_weight.data_ptr<input_t>(),
          freq_cis.data_ptr<float>(),
          out_loc_i64.data_ptr<int64_t>(),
          kvcache.data_ptr<uint8_t>(),
          plan_d,
          plan_c,
          static_cast<uint32_t>(num_tokens),
          static_cast<uint32_t>(compress_ratio),
          page_size,
          page_bits,
          kvcache.stride(0),
          static_cast<float>(norm_eps),
          is_decode,
          use_bf16_store,
          smem_data,
          smem_red,
          smem_group_inv,
          smem_group_ue8,
      };

      const uint32_t global_size = static_cast<uint32_t>(num_tokens) * kFlashMLALocalSize;
      cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kFlashMLALocalSize)), kernel);
    });
  });
}

}  // namespace at::native::xpu
