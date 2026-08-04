#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "Compress.h"
#include "Utils.h"

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

template <typename input_t>
struct FusedNormRopeIndexerKernel {
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

    const input_t* row_in = input + static_cast<int64_t>(group_id) * kHeadDimIndexer;

    float local_sum = 0.0f;
    for (int64_t i = tid; i < kHeadDimIndexer; i += kIndexerLocalSize) {
      const float x = to_float(row_in[i]);
      local_sum += x * x;
      data[i] = x;
    }

    red[tid] = local_sum;
    item.barrier(sycl::access::fence_space::local_space);

    for (uint32_t stride = kIndexerLocalSize / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        red[tid] += red[tid + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    const float norm_factor = sycl::rsqrt(red[0] / static_cast<float>(kHeadDimIndexer) + eps);
    for (int64_t i = tid; i < kHeadDimIndexer; i += kIndexerLocalSize) {
      data[i] = data[i] * norm_factor * to_float(weight[i]);
    }
    item.barrier(sycl::access::fence_space::local_space);

    const float* freq = freqs_cis + position * kRopeDim;
    for (int64_t p = tid; p < kRopeDim / 2; p += kIndexerLocalSize) {
      const int64_t base = 64 + 2 * p;
      const float xr = data[base + 0];
      const float xi = data[base + 1];
      const float fr = freq[2 * p + 0];
      const float fi = freq[2 * p + 1];
      data[base + 0] = xr * fr - xi * fi;
      data[base + 1] = xr * fi + xi * fr;
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (int64_t step = 1; step < kHeadDimIndexer; step <<= 1) {
      for (int64_t pair_id = tid; pair_id < kHeadDimIndexer / 2; pair_id += kIndexerLocalSize) {
        const int64_t block = (pair_id / step) * (2 * step);
        const int64_t off = pair_id % step;
        const int64_t a = block + off;
        const int64_t b = a + step;
        const float u = data[a];
        const float v = data[b];
        data[a] = u + v;
        data[b] = u - v;
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    // 1 / sqrt(128)
    constexpr float kHadamardScale = 0.08838834764831845f;
    for (int64_t i = tid; i < kHeadDimIndexer; i += kIndexerLocalSize) {
      data[i] *= kHadamardScale;
    }
    item.barrier(sycl::access::fence_space::local_space);

    const int64_t page = slot >> page_bits;
    const int64_t offset = slot & (page_size - 1);

    if (!use_fp4) {
      float local_max = 0.0f;
      for (int64_t i = tid; i < kHeadDimIndexer; i += kIndexerLocalSize) {
        local_max = sycl::fmax(local_max, sycl::fabs(data[i]));
      }
      red[tid] = local_max;
      item.barrier(sycl::access::fence_space::local_space);

      for (uint32_t stride = kIndexerLocalSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
          red[tid] = sycl::fmax(red[tid], red[tid + stride]);
        }
        item.barrier(sycl::access::fence_space::local_space);
      }

      const float scale = sycl::fmax(1.0e-4f, red[0]) / kFp8E4m3Max;
      const float inv_scale = 1.0f / scale;
      const int64_t value_base = page * page_bytes + offset * 128;
      const int64_t scale_base = page * page_bytes + 128 * page_size + offset * 4;

      for (int64_t i = tid; i < kHeadDimIndexer; i += kIndexerLocalSize) {
        const uint8_t q = cvt_float_to_fp8_e4m3(data[i] * inv_scale);
        kvcache[value_base + i] = q;
      }

      if (tid == 0) {
        const uint32_t bits = sycl::bit_cast<uint32_t>(scale);
        kvcache[scale_base + 0] = static_cast<uint8_t>(bits & 0xFFu);
        kvcache[scale_base + 1] = static_cast<uint8_t>((bits >> 8) & 0xFFu);
        kvcache[scale_base + 2] = static_cast<uint8_t>((bits >> 16) & 0xFFu);
        kvcache[scale_base + 3] = static_cast<uint8_t>((bits >> 24) & 0xFFu);
      }
      return;
    }

    for (int64_t g = 0; g < 4; ++g) {
      float g_local_max = 0.0f;
      for (int64_t j = tid; j < 32; j += kIndexerLocalSize) {
        const float v = sycl::fabs(data[g * 32 + j]);
        g_local_max = sycl::fmax(g_local_max, v);
      }
      red[tid] = g_local_max;
      item.barrier(sycl::access::fence_space::local_space);

      for (uint32_t stride = kIndexerLocalSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
          red[tid] = sycl::fmax(red[tid], red[tid + stride]);
        }
        item.barrier(sycl::access::fence_space::local_space);
      }

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

    for (int64_t i = tid; i < 64; i += kIndexerLocalSize) {
      const int64_t i0 = 2 * i;
      const int64_t i1 = i0 + 1;
      const int64_t g0 = i0 / 32;
      const int64_t g1 = i1 / 32;
      const uint8_t q0 = quant_fp4_e2m1(data[i0] * group_inv[g0]);
      const uint8_t q1 = quant_fp4_e2m1(data[i1] * group_inv[g1]);
      kvcache[value_base + i] = static_cast<uint8_t>((q0 & 0x0F) | ((q1 & 0x0F) << 4));
    }

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
  sycl::local_accessor<float, 1> smem_data;
  sycl::local_accessor<float, 1> smem_red;
  sycl::local_accessor<float, 1> smem_group_inv;
  sycl::local_accessor<uint8_t, 1> smem_group_ue8;
};

template <typename input_t>
struct FusedNormRopeFlashMLAKernel {
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

    const input_t* row_in = input + static_cast<int64_t>(group_id) * kHeadDimFlashMLA;

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

    red[tid] = local_sum;
    item.barrier(sycl::access::fence_space::local_space);

    for (uint32_t stride = kFlashMLALocalSize / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        red[tid] += red[tid + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    const float norm_factor = sycl::rsqrt(red[0] / static_cast<float>(kHeadDimFlashMLA) + eps);
    if (d0 < kHeadDimFlashMLA) {
      data[d0] = data[d0] * norm_factor * to_float(weight[d0]);
    }
    if (d1 < kHeadDimFlashMLA) {
      data[d1] = data[d1] * norm_factor * to_float(weight[d1]);
    }
    item.barrier(sycl::access::fence_space::local_space);

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

    for (int64_t g = 0; g < kNopeWarpsFlashMLA; ++g) {
      const int64_t start = g * kElemsPerNopeWarp;
      const int64_t end = start + kElemsPerNopeWarp;

      float g_local_max = 0.0f;
      if (d0 >= start && d0 < end) {
        g_local_max = sycl::fmax(g_local_max, sycl::fabs(data[d0]));
      }
      if (d1 >= start && d1 < end) {
        g_local_max = sycl::fmax(g_local_max, sycl::fabs(data[d1]));
      }

      red[tid] = g_local_max;
      item.barrier(sycl::access::fence_space::local_space);

      for (uint32_t stride = kFlashMLALocalSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
          red[tid] = sycl::fmax(red[tid], red[tid + stride]);
        }
        item.barrier(sycl::access::fence_space::local_space);
      }

      if (tid == 0) {
        const float scale_raw = sycl::fmax(1.0e-4f, red[0]) / kFp8E4m3Max;
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

    for (int64_t i = tid; i < kNopeDimFlashMLA; i += kFlashMLALocalSize) {
      const int64_t group = i / kElemsPerNopeWarp;
      const uint8_t q = cvt_float_to_fp8_e4m3(data[i] * group_inv[group]);
      kvcache[value_base + i] = q;
    }

    for (int64_t i = tid; i < kRopeDim; i += kFlashMLALocalSize) {
      const uint16_t bits = float_to_bf16_bits(data[kNopeDimFlashMLA + i]);
      const int64_t byte_off = value_base + kNopeDimFlashMLA + i * 2;
      kvcache[byte_off + 0] = static_cast<uint8_t>(bits & 0xFFu);
      kvcache[byte_off + 1] = static_cast<uint8_t>((bits >> 8) & 0xFFu);
    }

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
  sycl::local_accessor<float, 1> smem_data;
  sycl::local_accessor<float, 1> smem_red;
  sycl::local_accessor<float, 1> smem_group_inv;
  sycl::local_accessor<uint8_t, 1> smem_group_ue8;
};

}  // namespace

void fused_norm_rope_store(
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
    bool use_fp4) {
  TORCH_CHECK(input.is_xpu() && input.dim() == 2 && input.is_contiguous(), "input must be contiguous [N, head_dim] XPU tensor");
  TORCH_CHECK(
      plan.is_xpu() && plan.dtype() == torch::kUInt8 && plan.dim() == 2 && plan.is_contiguous(),
      "plan must be contiguous [N, 16] uint8 XPU tensor");
  TORCH_CHECK(
      norm_weight.is_xpu() && norm_weight.dim() == 1 && norm_weight.is_contiguous(),
      "norm_weight must be contiguous [head_dim] XPU tensor");
  TORCH_CHECK(
      freq_cis.is_xpu() && freq_cis.dtype() == torch::kFloat && freq_cis.dim() == 2 && freq_cis.is_contiguous(),
      "freq_cis must be contiguous [max_pos, 64] float32 XPU tensor");
  TORCH_CHECK(out_loc.is_xpu() && out_loc.dim() == 1 && out_loc.is_contiguous(), "out_loc must be contiguous [M] XPU tensor");
  TORCH_CHECK(
      kvcache.is_xpu() && kvcache.dtype() == torch::kUInt8 && kvcache.dim() == 2 && kvcache.is_contiguous(),
      "kvcache must be contiguous [num_pages, page_bytes] uint8 XPU tensor");

  const int64_t num_tokens = input.size(0);
  const int64_t head_dim = input.size(1);

  TORCH_CHECK(head_dim == kHeadDimIndexer || head_dim == kHeadDimFlashMLA, "head_dim must be 128 or 512, got ", head_dim);
  TORCH_CHECK(norm_weight.size(0) == head_dim, "norm_weight size must equal head_dim");
  TORCH_CHECK(freq_cis.size(1) == kRopeDim, "freq_cis last dim must be 64");
  TORCH_CHECK(plan.size(0) == num_tokens && plan.size(1) == static_cast<int64_t>(sizeof(DecodePlan)), "plan must be [N, 16]");
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

  const int64_t expected_page_bytes = (head_dim == kHeadDimIndexer)
      ? ((use_fp4 ? 68 : 132) * page_size)
      : flashmla_page_bytes(page_size);
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
            smem_data,
            smem_red,
            smem_group_inv,
            smem_group_ue8,
        };

        const uint32_t global_size = static_cast<uint32_t>(num_tokens) * kIndexerLocalSize;
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kIndexerLocalSize)), kernel);
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
          smem_data,
          smem_red,
          smem_group_inv,
          smem_group_ue8,
      };

      const uint32_t global_size = static_cast<uint32_t>(num_tokens) * kFlashMLALocalSize;
      cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(global_size), sycl::range<1>(kFlashMLALocalSize)), kernel);
    });
  });
}

}  // namespace at::native::xpu
