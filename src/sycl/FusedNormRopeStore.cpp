#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "Compress.h"
#include "QuantUtils.h"
#include "Utils.h"
#include "cutlass/float8.h"
#include "sgl_kernel_export.h"

using cutlass::float_e4m3_t;

namespace at::native::xpu {

namespace {

constexpr uint32_t kSubGroupSize = 16;
constexpr int64_t kRopeDim = 64;
constexpr float kFp8E4m3Max = 448.0f;

// Indexer variant constants
constexpr int64_t kIndexerHeadDim = 128;
constexpr uint32_t kIndexerBlockSize = kIndexerHeadDim / 2;  // 64
constexpr uint32_t kFp4GroupDim = 32;
constexpr uint32_t kFp4Groups = kIndexerHeadDim / kFp4GroupDim;  // 4

// FlashMLA variant constants
constexpr int64_t kFlashMLAHeadDim = 512;
constexpr uint32_t kFlashMLABlockSize = kFlashMLAHeadDim / 2;      // 256
constexpr int64_t kFlashMLANopeDim = kFlashMLAHeadDim - kRopeDim;  // 448
constexpr int64_t kFlashMLANopeSgs = 7;
constexpr int64_t kFlashMLAElemsPerNopeSg = kFlashMLANopeDim / kFlashMLANopeSgs;  // 64

inline int64_t flashmla_page_bytes(int64_t page_size) {
  // FlashMLA per-token cache layout: NoPE FP8 + RoPE BF16 + scales.
  constexpr int64_t kFlashMLAValueBytes = kFlashMLANopeDim + kRopeDim * 2;  // 576
  constexpr int64_t kFlashMLAScaleBytes = 8;
  constexpr int64_t kFlashMLAPerTokenBytes = kFlashMLAValueBytes + kFlashMLAScaleBytes;  // 584
  return ((kFlashMLAPerTokenBytes * page_size + kFlashMLAValueBytes - 1) / kFlashMLAValueBytes) * kFlashMLAValueBytes;
}

inline int64_t log2_i64(int64_t v) {
  int64_t bits = 0;
  while ((1LL << bits) < v) {
    ++bits;
  }
  return bits;
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
  const float_e4m3_t fp8 = static_cast<float_e4m3_t>(x);
  return sycl::bit_cast<uint8_t>(fp8);
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

// Reduces local_val across the work-group; result in red[0]. num_sg = local_size / 16.
inline void local_reduce_sum_sg16(
    float* red, sycl::nd_item<1> item, sycl::sub_group sg, uint32_t lid, float local_val, uint32_t num_sg) {
  const uint32_t lane_id = lid & (kSubGroupSize - 1u);
  const uint32_t sg_id = lid / kSubGroupSize;
  const float sg_sum = subgroup_xor_reduce_sum_16(sg, local_val);
  if (lane_id == 0) {
    red[sg_id] = sg_sum;
  }
  item.barrier(sycl::access::fence_space::local_space);

  if (sg_id == 0) {
    const float block_sum = subgroup_xor_reduce_sum_16(sg, (lane_id < num_sg) ? red[lane_id] : 0.0f);
    if (lane_id == 0) {
      red[0] = block_sum;
    }
  }
  item.barrier(sycl::access::fence_space::local_space);
}

inline void local_reduce_max_64_sg16(float* red, sycl::nd_item<1> item, sycl::sub_group sg, uint32_t lid) {
  const uint32_t lane_id = lid & (kSubGroupSize - 1u);
  const uint32_t sg_id = lid / kSubGroupSize;
  const float sg_max = subgroup_xor_reduce_max_16(sg, red[lid]);
  if (lane_id == 0) {
    red[sg_id] = sg_max;
  }
  item.barrier(sycl::access::fence_space::local_space);

  if (sg_id == 0) {
    const float block_max =
        subgroup_xor_reduce_max_16(sg, (lane_id < kIndexerBlockSize / kSubGroupSize) ? red[lane_id] : 0.0f);
    if (lane_id == 0) {
      red[0] = block_max;
    }
  }
  item.barrier(sycl::access::fence_space::local_space);
}

struct PlanSlot {
  int64_t position;
  int64_t slot;
  bool active;
};

inline PlanSlot read_plan_slot(
    const DecodePlan* plan_d,
    const CompressPlan* plan_c,
    const int64_t* out_loc,
    uint32_t group_id,
    uint32_t compress_ratio,
    bool is_decode) {
  PlanSlot r{0, -1, false};
  if (is_decode) {
    const DecodePlan plan = plan_d[group_id];
    if (plan.seq_len != 0 && plan.seq_len % compress_ratio == 0) {
      r.position = static_cast<int64_t>(plan.seq_len) - static_cast<int64_t>(compress_ratio);
      r.slot = out_loc[group_id];
      r.active = true;
    }
  } else {
    const CompressPlan plan = plan_c[group_id];
    if (!plan.is_invalid()) {
      r.position = static_cast<int64_t>(plan.seq_len) - static_cast<int64_t>(compress_ratio);
      r.slot = out_loc[static_cast<int64_t>(plan.ragged_id)];
      r.active = true;
    }
  }
  return r;
}

template <typename input_t>
struct FusedNormRopeIndexerKernel {
  // Indexer variant: head_dim=128, one token per work-group (64 threads).
  // Each lane_id handles 2 elements, covering all 128 dims per token.
  // Cache layout per token:
  //  - FP8 path: 128 value bytes + 4-byte fp32 scale
  //  - FP4 path: 64 packed bytes + 4 UE8M0 scale bytes
  [[sycl::reqd_sub_group_size(kSubGroupSize)]]
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t gid = static_cast<uint32_t>(item.get_group(0));
    const uint32_t lid = static_cast<uint32_t>(item.get_local_id(0));
    if (gid >= num_tokens) {
      return;
    }

    const auto ps = read_plan_slot(plan_d, plan_c, out_loc, gid, compress_ratio, is_decode);
    if (!ps.active || ps.slot < 0) {
      return;
    }
    float* data = smem_data.template get_multi_ptr<sycl::access::decorated::no>().get();
    float* red = smem_red.template get_multi_ptr<sycl::access::decorated::no>().get();
    float* group_inv = smem_group_inv.template get_multi_ptr<sycl::access::decorated::no>().get();
    uint8_t* group_ue8 = smem_group_ue8.template get_multi_ptr<sycl::access::decorated::no>().get();
    auto sg = item.get_sub_group();

    const input_t* row_in = input + static_cast<int64_t>(gid) * kIndexerHeadDim;

    const int64_t d0 = static_cast<int64_t>(lid);
    const int64_t d1 = d0 + static_cast<int64_t>(kIndexerBlockSize);

    // Part 1: RMSNorm over the full 128-d vector.
    data[d0] = static_cast<float>(row_in[d0]);
    data[d1] = static_cast<float>(row_in[d1]);
    const float local_sum = data[d0] * data[d0] + data[d1] * data[d1];
    local_reduce_sum_sg16(red, item, sg, lid, local_sum, kIndexerBlockSize / kSubGroupSize);

    const float norm_factor = sycl::rsqrt(red[0] / static_cast<float>(kIndexerHeadDim) + eps);
    data[d0] = data[d0] * norm_factor * static_cast<float>(weight[d0]);
    data[d1] = data[d1] * norm_factor * static_cast<float>(weight[d1]);
    item.barrier(sycl::access::fence_space::local_space);

    // Part 2: RoPE on the tail 64 dims.
    const float* freq = freqs_cis + ps.position * kRopeDim;
    if (lid < static_cast<uint32_t>(kRopeDim / 2)) {
      const int64_t p = static_cast<int64_t>(lid);
      const int64_t base = 64 + 2 * p;
      const float tmp = data[base + 0];
      data[base + 0] = tmp * freq[2 * p + 0] - data[base + 1] * freq[2 * p + 1];
      data[base + 1] = tmp * freq[2 * p + 1] + data[base + 1] * freq[2 * p + 0];
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Part 3: hadamard transform.
    // 1) register-local pair butterfly (a+b, a-b)
    float h0 = data[d0] + data[d1];
    float h1 = data[d0] - data[d1];

    // 2) intra-subgroup XOR butterflies (mask 1/2/4/8)
    const uint32_t lane_id = lid & 0xFu;
    for (uint32_t mask = 1; mask <= 8; mask <<= 1) {
      const float o0 = sycl::permute_group_by_xor(sg, h0, mask);
      const float o1 = sycl::permute_group_by_xor(sg, h1, mask);
      h0 = (lane_id & mask) ? (o0 - h0) : (h0 + o0);
      h1 = (lane_id & mask) ? (o1 - h1) : (h1 + o1);
    }

    data[d0] = h0;
    data[d1] = h1;
    item.barrier(sycl::access::fence_space::local_space);

    // 3) cross-subgroup butterflies (mask 16/32) via local memory
    for (uint32_t mask = kSubGroupSize; mask <= 32; mask <<= 1) {
      const uint32_t peer = lid ^ mask;
      const float o0 = data[static_cast<int64_t>(peer)];
      const float o1 = data[static_cast<int64_t>(peer) + kIndexerBlockSize];
      h0 = (lid & mask) ? (o0 - h0) : (h0 + o0);
      h1 = (lid & mask) ? (o1 - h1) : (h1 + o1);
      data[d0] = h0;
      data[d1] = h1;
      item.barrier(sycl::access::fence_space::local_space);
    }

    constexpr float kHadamardScale = 0.08838834764831845f;  // 1 / sqrt(128)
    data[d0] = h0 * kHadamardScale;
    data[d1] = h1 * kHadamardScale;
    item.barrier(sycl::access::fence_space::local_space);

    const int64_t page = ps.slot >> page_bits;
    const int64_t offset = ps.slot & (page_size - 1);

    // Part 4a: FP8 store. For preshuffle_size>0, values are written in tiled
    // order to match the pre-shuffled consumer layout.
    if (!use_fp4) {
      float local_max = sycl::fmax(sycl::fabs(data[d0]), sycl::fabs(data[d1]));
      red[lid] = local_max;
      local_reduce_max_64_sg16(red, item, sg, lid);

      const float scale = sycl::fmax(1.0e-4f, red[0]) / kFp8E4m3Max;
      const float inv_scale = 1.0f / scale;
      const int64_t scale_base =
          page * page_bytes + kIndexerHeadDim * page_size + offset * static_cast<int64_t>(sizeof(float));
      for (int64_t pair = lid; pair < kIndexerHeadDim / 2; pair += kIndexerBlockSize) {
        const int64_t i0 = 2 * pair;
        const int64_t i1 = i0 + 1;
        const uint8_t q0 = cvt_float_to_fp8_e4m3(data[i0] * inv_scale);
        const uint8_t q1 = cvt_float_to_fp8_e4m3(data[i1] * inv_scale);

        if (preshuffle_size == 0) {
          const int64_t value_base = page * page_bytes + offset * kIndexerHeadDim;
          reinterpret_cast<uint16_t*>(kvcache + value_base)[pair] = pack_u8x2(q0, q1);
          continue;
        }

        const int64_t token_tile_id = offset / preshuffle_size;
        const int64_t token_in_tile = offset % preshuffle_size;

        const int64_t col_tile_id0 = i0 / preshuffle_size;
        const int64_t col_in_tile0 = i0 % preshuffle_size;
        const int64_t value_offset0 = token_tile_id * (preshuffle_size * kIndexerHeadDim) +
                                      col_tile_id0 * (preshuffle_size * preshuffle_size) +
                                      token_in_tile * preshuffle_size + col_in_tile0;
        kvcache[page * page_bytes + value_offset0] = q0;

        const int64_t col_tile_id1 = i1 / preshuffle_size;
        const int64_t col_in_tile1 = i1 % preshuffle_size;
        const int64_t value_offset1 = token_tile_id * (preshuffle_size * kIndexerHeadDim) +
                                      col_tile_id1 * (preshuffle_size * preshuffle_size) +
                                      token_in_tile * preshuffle_size + col_in_tile1;
        kvcache[page * page_bytes + value_offset1] = q1;
      }

      if (lid == 0) {
        reinterpret_cast<uint32_t*>(kvcache + scale_base)[0] = sycl::bit_cast<uint32_t>(scale);
      }
      return;
    }

    // Part 4b: FP4 store (kFp4Groups groups x kFp4GroupDim dims), each group owns one UE8M0 scale.
    for (int64_t g = 0; g < kFp4Groups; ++g) {
      float g_local_max = 0.0f;
      if (lid < kFp4GroupDim) {
        g_local_max = sycl::fabs(data[g * static_cast<int64_t>(kFp4GroupDim) + static_cast<int64_t>(lid)]);
      }
      red[lid] = g_local_max;
      local_reduce_max_64_sg16(red, item, sg, lid);

      if (lid == 0) {
        const float scale_raw = sycl::fmax(1.0e-4f, red[0]) / 6.0f;
        const uint8_t ue8 = castToUE8M0(scale_raw);
        group_ue8[g] = ue8;
        group_inv[g] = invScaleUE8M0(ue8);
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    const int64_t value_base = page * page_bytes + offset * (kIndexerHeadDim / 2);
    const int64_t scale_base = page * page_bytes + (kIndexerHeadDim / 2) * page_size + offset * kFp4Groups;

    const int64_t i = static_cast<int64_t>(lid);
    const int64_t i0 = 2 * i;
    const int64_t i1 = i0 + 1;
    const int64_t g0 = i0 / static_cast<int64_t>(kFp4GroupDim);
    const int64_t g1 = i1 / static_cast<int64_t>(kFp4GroupDim);
    const uint8_t q0 = quant_fp4_e2m1(data[i0] * group_inv[g0]);
    const uint8_t q1 = quant_fp4_e2m1(data[i1] * group_inv[g1]);
    kvcache[value_base + i] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));

    if (lid < kFp4Groups) {
      kvcache[scale_base + lid] = group_ue8[lid];
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
  // Each lane_id handles 2 elements, covering all 512 dims per token.
  [[sycl::reqd_sub_group_size(kSubGroupSize)]]
  void operator()(sycl::nd_item<1> item) const {
    const uint32_t gid = static_cast<uint32_t>(item.get_group(0));
    const uint32_t lid = static_cast<uint32_t>(item.get_local_id(0));
    if (gid >= num_tokens) {
      return;
    }

    const auto ps = read_plan_slot(plan_d, plan_c, out_loc, gid, compress_ratio, is_decode);
    if (!ps.active || ps.slot < 0) {
      return;
    }
    float* data = smem_data.template get_multi_ptr<sycl::access::decorated::no>().get();
    float* red = smem_red.template get_multi_ptr<sycl::access::decorated::no>().get();
    float* group_inv = smem_group_inv.template get_multi_ptr<sycl::access::decorated::no>().get();
    uint8_t* group_ue8 = smem_group_ue8.template get_multi_ptr<sycl::access::decorated::no>().get();
    auto sg = item.get_sub_group();

    const input_t* row_in = input + static_cast<int64_t>(gid) * kFlashMLAHeadDim;

    // Part 1: RMSNorm over head_dim=512 using subgroup partial reductions.
    float local_sum = 0.0f;
    const int64_t d0 = static_cast<int64_t>(lid) * 2;
    const int64_t d1 = d0 + 1;
    if (d0 < kFlashMLAHeadDim) {
      const float x0 = static_cast<float>(row_in[d0]);
      data[d0] = x0;
      local_sum += x0 * x0;
    }
    if (d1 < kFlashMLAHeadDim) {
      const float x1 = static_cast<float>(row_in[d1]);
      data[d1] = x1;
      local_sum += x1 * x1;
    }

    const uint32_t lane_id = lid & (kSubGroupSize - 1u);
    const uint32_t sg_id = lid / kSubGroupSize;
    local_reduce_sum_sg16(red, item, sg, lid, local_sum, kFlashMLABlockSize / kSubGroupSize);

    const float norm_factor = sycl::rsqrt(red[0] / static_cast<float>(kFlashMLAHeadDim) + eps);
    if (d0 < kFlashMLAHeadDim) {
      data[d0] = data[d0] * norm_factor * static_cast<float>(weight[d0]);
    }
    if (d1 < kFlashMLAHeadDim) {
      data[d1] = data[d1] * norm_factor * static_cast<float>(weight[d1]);
    }
    item.barrier(sycl::access::fence_space::local_space);

    // Part 2: RoPE on the tail 64 dims.
    const float* freq = freqs_cis + ps.position * kRopeDim;
    for (int64_t p = lid; p < kRopeDim / 2; p += kFlashMLABlockSize) {
      const int64_t base = kFlashMLANopeDim + 2 * p;
      const float tmp = data[base + 0];
      data[base + 0] = tmp * freq[2 * p + 0] - data[base + 1] * freq[2 * p + 1];
      data[base + 1] = tmp * freq[2 * p + 1] + data[base + 1] * freq[2 * p + 0];
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (use_bf16_store) {
      // Optional mode: write the whole 512-d output as plain BF16.
      const int64_t page = ps.slot >> page_bits;
      const int64_t offset = ps.slot & (page_size - 1);
      const int64_t value_base = page * page_bytes + offset * (kFlashMLAHeadDim * 2);
      uint16_t* value_ptr = reinterpret_cast<uint16_t*>(kvcache + value_base);
      if (d0 < kFlashMLAHeadDim) {
        value_ptr[d0] = float_to_bf16_bits(data[d0]);
      }
      if (d1 < kFlashMLAHeadDim) {
        value_ptr[d1] = float_to_bf16_bits(data[d1]);
      }
      return;
    }

    const int64_t g_d0 = (d0 < kFlashMLANopeDim) ? (d0 / kFlashMLAElemsPerNopeSg) : -1;
    const int64_t g_d1 = (d1 < kFlashMLANopeDim) ? (d1 / kFlashMLAElemsPerNopeSg) : -1;

    // Part 3: NoPE FP8 quantization. Each 64-d group gets one UE8M0 scale.
#pragma unroll
    for (int64_t g = 0; g < kFlashMLANopeSgs; ++g) {
      float g_local_max = 0.0f;
      if (g_d0 == g) {
        g_local_max = sycl::fabs(data[d0]);
      }
      if (g_d1 == g) {
        g_local_max = sycl::fmax(g_local_max, sycl::fabs(data[d1]));
      }

      const float sg_max = subgroup_xor_reduce_max_16(sg, g_local_max);
      if (lane_id == 0) {
        red[sg_id] = sg_max;
      }
      item.barrier(sycl::access::fence_space::local_space);

      if (lid == 0) {
        const uint32_t sg0 = static_cast<uint32_t>(g * 2);
        const float abs_max = sycl::fmax(red[sg0], red[sg0 + 1]);
        const float scale_raw = sycl::fmax(1.0e-4f, abs_max) / kFp8E4m3Max;
        const uint8_t ue8 = castToUE8M0(scale_raw);
        group_ue8[g] = ue8;
        group_inv[g] = invScaleUE8M0(ue8);
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    const int64_t page = ps.slot >> page_bits;
    const int64_t offset = ps.slot & (page_size - 1);
    const int64_t value_base = page * page_bytes + offset * 576;
    const int64_t scale_base = page * page_bytes + 576 * page_size + offset * 8;

    // NoPE values: 448 FP8 bytes packed as uint16 pairs.
    uint16_t* nope_ptr = reinterpret_cast<uint16_t*>(kvcache + value_base);
    if (lid < static_cast<uint32_t>(kFlashMLANopeDim / 2)) {
      const int64_t pair = static_cast<int64_t>(lid);
      const int64_t i0 = 2 * pair;
      const int64_t i1 = i0 + 1;
      const int64_t g0 = i0 / kFlashMLAElemsPerNopeSg;
      const int64_t g1 = i1 / kFlashMLAElemsPerNopeSg;
      const uint8_t q0 = cvt_float_to_fp8_e4m3(data[i0] * group_inv[g0]);
      const uint8_t q1 = cvt_float_to_fp8_e4m3(data[i1] * group_inv[g1]);
      nope_ptr[pair] = pack_u8x2(q0, q1);
    }

    // RoPE tail: 64 BF16 values.
    uint16_t* rope_ptr = reinterpret_cast<uint16_t*>(kvcache + value_base + kFlashMLANopeDim);
    if (lid < static_cast<uint32_t>(kRopeDim)) {
      const int64_t i = static_cast<int64_t>(lid);
      rope_ptr[i] = float_to_bf16_bits(data[kFlashMLANopeDim + i]);
    }

    // Scale region: first 7 bytes are valid scales (one per NoPE group).
    if (lid < static_cast<uint32_t>(kFlashMLANopeSgs)) {
      kvcache[scale_base + lid] = group_ue8[lid];
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
  CHECK_INPUT(input);
  CHECK_DIM(2, input);
  CHECK_INPUT(plan);
  CHECK_DIM(2, plan);
  CHECK_EQ(plan.dtype(), torch::kUInt8);
  CHECK_INPUT(norm_weight);
  CHECK_DIM(1, norm_weight);
  CHECK_INPUT(freq_cis);
  CHECK_DIM(2, freq_cis);
  CHECK_EQ(freq_cis.dtype(), torch::kFloat);
  CHECK_INPUT(out_loc);
  CHECK_DIM(1, out_loc);
  CHECK_INPUT(kvcache);
  CHECK_DIM(2, kvcache);
  CHECK_EQ(kvcache.dtype(), torch::kUInt8);

  const int64_t num_tokens = input.size(0);
  const int64_t head_dim = input.size(1);

  TORCH_CHECK(
      head_dim == kIndexerHeadDim || head_dim == kFlashMLAHeadDim, "head_dim must be 128 or 512, got ", head_dim);
  TORCH_CHECK(norm_weight.size(0) == head_dim, "norm_weight size must equal head_dim");
  TORCH_CHECK(freq_cis.size(1) == kRopeDim, "freq_cis last dim must be 64");
  TORCH_CHECK(
      plan.size(0) == num_tokens && plan.size(1) == static_cast<int64_t>(sizeof(DecodePlan)), "plan must be [N, 16]");
  TORCH_CHECK(compress_ratio > 0, "compress_ratio must be > 0");
  TORCH_CHECK(page_size > 0 && (page_size & (page_size - 1)) == 0, "page_size must be power of 2");

  TORCH_CHECK(input.scalar_type() == norm_weight.scalar_type(), "input and norm_weight dtypes must match");
  TORCH_CHECK(
      input.scalar_type() == at::ScalarType::Half || input.scalar_type() == at::ScalarType::BFloat16 ||
          input.scalar_type() == at::ScalarType::Float,
      "input dtype must be fp16/bf16/fp32");

  if (use_fp4) {
    TORCH_CHECK(head_dim == kIndexerHeadDim, "use_fp4 is only supported for head_dim=128");
  }

  TORCH_CHECK(preshuffle_size >= 0, "preshuffle_size must be >= 0");
  if (preshuffle_size > 0) {
    TORCH_CHECK(head_dim == kIndexerHeadDim, "preshuffle_size is only supported for head_dim=128");
    TORCH_CHECK(!use_fp4, "preshuffle_size is not supported with use_fp4=True");
    TORCH_CHECK(preshuffle_size % 2 == 0, "preshuffle_size must be even");
    TORCH_CHECK(kIndexerHeadDim % preshuffle_size == 0, "head_dim(128) must be divisible by preshuffle_size");
    TORCH_CHECK(page_size % preshuffle_size == 0, "page_size must be divisible by preshuffle_size");
  }

  if (use_bf16_store) {
    TORCH_CHECK(head_dim == kFlashMLAHeadDim, "use_bf16_store is only supported for head_dim=512");
    TORCH_CHECK(!use_fp4, "use_bf16_store is not supported with use_fp4=True");
  }

  const int64_t expected_page_bytes =
      (head_dim == kIndexerHeadDim)
          ? ((use_fp4 ? 68 : 132) * page_size)
          : (use_bf16_store ? (kFlashMLAHeadDim * 2 * page_size) : flashmla_page_bytes(page_size));
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

    if (head_dim == kIndexerHeadDim) {
      queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> smem_data(sycl::range<1>(kIndexerHeadDim), cgh);
        sycl::local_accessor<float, 1> smem_red(sycl::range<1>(kIndexerBlockSize), cgh);
        sycl::local_accessor<float, 1> smem_group_inv(sycl::range<1>(kFp4Groups), cgh);
        sycl::local_accessor<uint8_t, 1> smem_group_ue8(sycl::range<1>(kFp4Groups), cgh);

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

        const uint32_t global_size = static_cast<uint32_t>(num_tokens) * kIndexerBlockSize;
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kIndexerBlockSize)), kernel);
      });
      return;
    }

    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> smem_data(sycl::range<1>(kFlashMLAHeadDim), cgh);
      sycl::local_accessor<float, 1> smem_red(sycl::range<1>(kFlashMLABlockSize), cgh);
      sycl::local_accessor<float, 1> smem_group_inv(sycl::range<1>(kFlashMLANopeSgs), cgh);
      sycl::local_accessor<uint8_t, 1> smem_group_ue8(sycl::range<1>(kFlashMLANopeSgs), cgh);

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

      const uint32_t global_size = static_cast<uint32_t>(num_tokens) * kFlashMLABlockSize;
      cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kFlashMLABlockSize)), kernel);
    });
  });
}

}  // namespace at::native::xpu
