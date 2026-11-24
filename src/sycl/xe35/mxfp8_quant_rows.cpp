/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief Row-major MXFP8 (E4M3 + UE8M0 per 32) activation quantization for the
           MoE grouped GEMMs on Xe3P, with an optional fused SiLU-and-mul.

    Replaces sgl_per_token_group_quant_8bit(group_size=32, scale_ue8m0=True) on
    the MoE path. That kernel maps one 32-element group onto a whole 16-lane
    subgroup with 8 elements per lane, so 12 of 16 lanes idle; at MoE prefill
    sizes it ran ~10x below bandwidth. Here a group is 4 lanes x 8 elements
    (one 16-byte load per lane), the amax reduction is two xor-shuffles, and
    every lane converts and stores its own 8 bytes.

    Scale semantics match the existing kernel: s = 2^ceil(log2(max(amax, 1e-10) / 448)),
    stored as the biased exponent, q = sat_e4m3(x / s). ceil(log2(.)) is computed
    exactly from the float bits (the old kernel used sycl::log2, which can round
    a value just above a power of two down to the integer). x / s is exact, and
    because |x / s| <= 448 and x carries at most 11 significant bits, the
    float -> half step is exact too, so the single rounding is half -> E4M3 RNE.

    fuse_silu_and_mul: input is (M, 2N) [gate | up] in bf16 / fp16 / fp32 and the
    output is quantize(bf16(silu(gate) * up)) of shape (M, N). The bf16 rounding
    of the activation (and of fp32 inputs) reproduces the unfused
    silu_and_mul(c1.to(bf16)) -> quantize sequence it replaces.
*/

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/oneapi/experimental/float_8bit/types.hpp>
#include <sycl/sycl.hpp>

#include <algorithm>

namespace at::native::xpu {
namespace {

constexpr int kMxGroup = 32;
constexpr int kElemsPerLane = 8;
constexpr int kLanesPerGroup = kMxGroup / kElemsPerLane;  // 4
constexpr int kSubGroup = 16;
constexpr int kWorkGroup = 256;

inline float bf16_bits_to_float(uint16_t b) {
  return sycl::bit_cast<float>(static_cast<uint32_t>(b) << 16);
}

// Round-to-nearest-even float -> bf16 -> float (finite inputs).
inline float round_to_bf16(float x) {
  uint32_t u = sycl::bit_cast<uint32_t>(x);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return sycl::bit_cast<float>(u & 0xFFFF0000u);
}

// Rounding of the activation to the dtype the unfused path materializes:
// fp16 for fp16 inputs, bf16 otherwise.
template <typename InT>
inline float round_act(float x) {
  if constexpr (std::is_same_v<InT, sycl::half>) {
    return static_cast<float>(static_cast<sycl::half>(x));
  } else {
    return round_to_bf16(x);
  }
}

template <typename T>
struct Loader;

template <>
struct Loader<sycl::ext::oneapi::bfloat16> {
  static void load8(const void* base, int64_t idx, float (&v)[kElemsPerLane]) {
    auto w = *reinterpret_cast<const sycl::uint4*>(reinterpret_cast<const uint16_t*>(base) + idx);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint32_t d = w[i];
      v[2 * i] = sycl::bit_cast<float>(d << 16);
      v[2 * i + 1] = sycl::bit_cast<float>(d & 0xFFFF0000u);
    }
  }
};

template <>
struct Loader<sycl::half> {
  static void load8(const void* base, int64_t idx, float (&v)[kElemsPerLane]) {
    auto w = *reinterpret_cast<const sycl::vec<sycl::half, 8>*>(reinterpret_cast<const sycl::half*>(base) + idx);
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) v[i] = static_cast<float>(w[i]);
  }
};

template <>
struct Loader<float> {
  static void load8(const void* base, int64_t idx, float (&v)[kElemsPerLane]) {
    auto p = reinterpret_cast<const sycl::float4*>(reinterpret_cast<const float*>(base) + idx);
    sycl::float4 a = p[0], b = p[1];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      v[i] = a[i];
      v[4 + i] = b[i];
    }
  }
};

template <typename InT, bool FuseSilu>
class Mxfp8QuantRowsKernel {
 public:
  Mxfp8QuantRowsKernel(
      const void* in,
      uint8_t* out_q,
      uint8_t* out_s,
      const int32_t* src2dst,
      int topk,
      int lanes_per_row,
      int64_t k_out,
      int64_t in_row_stride)
      : in_(in),
        out_q_(out_q),
        out_s_(out_s),
        src2dst_(src2dst),
        topk_(topk),
        aligned_scales_(k_out % (4 * kMxGroup) == 0),
        lanes_per_row_(lanes_per_row),
        k_out_(k_out),
        in_row_stride_(in_row_stride) {}

  // 2D launch: dim 0 = row, dim 1 = 8-element lane slot within the row. This
  // keeps the 64-bit index math (emulated division on Xe) out of the kernel.
  [[sycl::reqd_sub_group_size(kSubGroup)]] void operator()(sycl::nd_item<2> it) const {
    const int64_t row = static_cast<int64_t>(it.get_global_id(0));
    const int slot = static_cast<int>(it.get_global_id(1));
    // lanes_per_row is a multiple of 4, so a 4-lane quant group is either
    // fully live or fully dead; dead groups still join the shuffles below and
    // read a valid address (slot 0 of the row), but never store.
    const bool live = slot < lanes_per_row_;
    const int64_t col = static_cast<int64_t>(live ? slot : 0) * kElemsPerLane;
    const int64_t flat = row * k_out_ + col;

    float v[kElemsPerLane];
    if constexpr (FuseSilu) {
      float g[kElemsPerLane], u[kElemsPerLane];
      Loader<InT>::load8(in_, row * in_row_stride_ + col, g);
      Loader<InT>::load8(in_, row * in_row_stride_ + k_out_ + col, u);
#pragma unroll
      for (int i = 0; i < kElemsPerLane; ++i) {
        float a = g[i], b = u[i];
        if constexpr (std::is_same_v<InT, float>) {
          a = round_to_bf16(a);
          b = round_to_bf16(b);
        }
        // native exp / reciprocal: a few-ulp fp32 error is far below the
        // bf16 rounding applied next (the generic silu_and_mul kernel is
        // itself off by >= 1 bf16 ulp on ~27% of outputs vs. float64).
        v[i] = round_act<InT>(a * sycl::native::recip(1.0f + sycl::native::exp(-a)) * b);
      }
    } else {
      Loader<InT>::load8(in_, row * in_row_stride_ + col, v);
      if constexpr (std::is_same_v<InT, float>) {
#pragma unroll
        for (int i = 0; i < kElemsPerLane; ++i) v[i] = round_to_bf16(v[i]);
      }
    }

    float amax = 1e-10f;
#pragma unroll
    for (int i = 0; i < kElemsPerLane; ++i) amax = sycl::fmax(amax, sycl::fabs(v[i]));
    auto sg = it.get_sub_group();
    amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 1));
    amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, 2));

    // Exact ceil(log2(max(amax / 448, 1e-10))) without the (correctly rounded,
    // hence expensive) fp32 divide. With amax = m * 2^E, m in [1, 2), and
    // 448 = 1.75 * 2^8: amax / 448 = (m / 1.75) * 2^(E-8), where m / 1.75 is in
    // (0.5, 1] for m <= 1.75 and in (1, 2) above, so the ceiling is E - 8 or
    // E - 7. The eps floor (ceil(log2(1e-10)) = -33) becomes a max. amax >= 1e-10
    // is always a normal float.
    const uint32_t ab = sycl::bit_cast<uint32_t>(amax);
    const int e = sycl::max(
        static_cast<int>(ab >> 23) - 127 - 8 + ((ab & 0x7FFFFFu) > 0x600000u ? 1 : 0), -33);
    const float inv_s = sycl::bit_cast<float>(static_cast<uint32_t>(127 - e) << 23);

    // Collect the subgroup's four scale bytes (groups owned by lanes 0, 4, 8,
    // 12) into one dword: single-byte stores from every fourth lane cost more
    // than the rest of the kernel combined.
    const uint32_t sbyte = static_cast<uint32_t>(e + 127);
    const uint32_t s1 = sycl::select_from_group(sg, sbyte, 4);
    const uint32_t s2 = sycl::select_from_group(sg, sbyte, 8);
    const uint32_t s3 = sycl::select_from_group(sg, sbyte, 12);
    const uint32_t s0 = sycl::select_from_group(sg, sbyte, 0);

    if (!live) return;
    uint32_t packed[2] = {0u, 0u};
#ifdef __SYCL_DEVICE_ONLY__
#pragma unroll
    for (int i = 0; i < kElemsPerLane; i += 2) {
      sycl::detail::float16_vec2 h{static_cast<_Float16>(v[i] * inv_s), static_cast<_Float16>(v[i + 1] * inv_s)};
      auto q = __builtin_spirv_ClampConvertFP16ToE4M3INTEL(h);
      packed[i / 4] |= (static_cast<uint32_t>(q[0]) | (static_cast<uint32_t>(q[1]) << 8)) << (8 * (i % 4));
    }
#endif
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const uint32_t s_word = s0 | (s1 << 8) | (s2 << 16) | (s3 << 24);
    // Plain: one output row per input row. Scatter: input row t goes to output
    // rows src2dst[t * topk + j] (quantize each token once, not topk times).
    const int n_dst = topk_ > 0 ? topk_ : 1;
    for (int j = 0; j < n_dst; ++j) {
      const int64_t orow = topk_ > 0 ? static_cast<int64_t>(src2dst_[row * topk_ + j]) : row;
      *reinterpret_cast<sycl::uint2*>(out_q_ + orow * k_out_ + col) = sycl::uint2{packed[0], packed[1]};
      const int64_t s_idx = orow * (k_out_ / kMxGroup) + col / kMxGroup;
      if (aligned_scales_) {
        // k_out % 128 == 0: the 4 groups of a subgroup are contiguous and
        // 4-byte aligned; slot is a multiple of 16 at lane 0 (work-groups are
        // multiples of the subgroup size), and dead groups only occur past the
        // row end, which an aligned row never has inside a subgroup.
        if (lane == 0) *reinterpret_cast<uint32_t*>(out_s_ + s_idx) = s_word;
      } else if ((slot & (kLanesPerGroup - 1)) == 0) {
        out_s_[s_idx] = static_cast<uint8_t>(sbyte);
      }
    }
  }

 private:
  const void* in_;
  uint8_t* out_q_;
  uint8_t* out_s_;
  const int32_t* src2dst_;
  int topk_;
  bool aligned_scales_;
  int lanes_per_row_;
  int64_t k_out_;
  int64_t in_row_stride_;
};

template <typename InT, bool FuseSilu>
void launch(
    const at::Tensor& input,
    at::Tensor& out_q,
    at::Tensor& out_s,
    const int32_t* src2dst,
    int topk,
    int64_t m,
    int64_t k_out) {
  if (m == 0) return;
  const int lanes_per_row = static_cast<int>(k_out / kElemsPerLane);
  // One work-group spans up to 256 lane slots of a row; long rows use several.
  const int wg = static_cast<int>(std::min<int64_t>(kWorkGroup, (lanes_per_row + kSubGroup - 1) / kSubGroup * kSubGroup));
  const int64_t cols = (lanes_per_row + wg - 1) / wg * wg;
  auto& q = at::xpu::getCurrentXPUStream().queue();
  Mxfp8QuantRowsKernel<InT, FuseSilu> k(
      input.data_ptr(),
      reinterpret_cast<uint8_t*>(out_q.data_ptr()),
      out_s.data_ptr<uint8_t>(),
      src2dst,
      topk,
      lanes_per_row,
      k_out,
      input.size(1));
  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(m, cols), sycl::range<2>(1, wg)), k);
  });
}

}  // namespace

void mxfp8_quant_rows(
    at::Tensor input,
    at::Tensor output_q,
    at::Tensor output_s,
    bool fuse_silu_and_mul,
    const std::optional<at::Tensor>& src2dst) {
  TORCH_CHECK(input.is_xpu() && output_q.is_xpu() && output_s.is_xpu(), "mxfp8_quant_rows: tensors must be on XPU");
  TORCH_CHECK(input.dim() == 2 && output_q.dim() == 2 && output_s.dim() == 2, "mxfp8_quant_rows: expects 2D tensors");
  TORCH_CHECK(
      input.is_contiguous() && output_q.is_contiguous() && output_s.is_contiguous(),
      "mxfp8_quant_rows: tensors must be contiguous");
  TORCH_CHECK(output_q.scalar_type() == at::kFloat8_e4m3fn, "mxfp8_quant_rows: output_q must be float8_e4m3fn");
  TORCH_CHECK(output_s.scalar_type() == at::kByte, "mxfp8_quant_rows: output_s must be uint8 (UE8M0)");
  const int64_t m = input.size(0);
  const int64_t k_out = output_q.size(1);
  const int64_t m_out = output_q.size(0);
  TORCH_CHECK(output_s.size(0) == m_out, "mxfp8_quant_rows: output_q / output_s row count mismatch");
  const int32_t* src2dst_ptr = nullptr;
  int topk = 0;
  if (src2dst.has_value()) {
    TORCH_CHECK(!fuse_silu_and_mul, "mxfp8_quant_rows: src2dst scatter is only supported without fused silu");
    TORCH_CHECK(
        src2dst->is_xpu() && src2dst->scalar_type() == at::kInt && src2dst->is_contiguous(),
        "mxfp8_quant_rows: src2dst must be a contiguous int32 XPU tensor");
    TORCH_CHECK(m > 0 && m_out % m == 0 && src2dst->numel() == m_out, "mxfp8_quant_rows: src2dst must have M*topk entries");
    topk = static_cast<int>(m_out / m);
    TORCH_CHECK(topk >= 1 && topk <= 64, "mxfp8_quant_rows: unsupported topk ", topk);
    src2dst_ptr = src2dst->data_ptr<int32_t>();
  } else {
    TORCH_CHECK(m_out == m, "mxfp8_quant_rows: row count mismatch");
  }
  TORCH_CHECK(k_out % kMxGroup == 0, "mxfp8_quant_rows: K must be a multiple of 32, got ", k_out);
  TORCH_CHECK(output_s.size(1) == k_out / kMxGroup, "mxfp8_quant_rows: output_s must be (M, K/32)");
  TORCH_CHECK(
      input.size(1) == (fuse_silu_and_mul ? 2 * k_out : k_out),
      "mxfp8_quant_rows: input must be (M, ",
      fuse_silu_and_mul ? "2*" : "",
      "K)");

  const auto dt = input.scalar_type();
  auto dispatch = [&](auto tag) {
    using T = decltype(tag);
    if (fuse_silu_and_mul)
      launch<T, true>(input, output_q, output_s, src2dst_ptr, topk, m, k_out);
    else
      launch<T, false>(input, output_q, output_s, src2dst_ptr, topk, m, k_out);
  };
  if (dt == at::kBFloat16) {
    dispatch(sycl::ext::oneapi::bfloat16{});
  } else if (dt == at::kHalf) {
    dispatch(sycl::half{});
  } else if (dt == at::kFloat) {
    dispatch(float{});
  } else {
    TORCH_CHECK(false, "mxfp8_quant_rows: input must be bf16, fp16 or fp32");
  }
}

}  // namespace at::native::xpu
