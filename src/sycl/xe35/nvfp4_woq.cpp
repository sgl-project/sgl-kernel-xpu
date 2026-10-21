// SPDX-License-Identifier: Apache-2.0
/*
 * NVFP4 weight-only-quantization (WoQ) support kernels.
 *
 * Xe3P has no native NVFP4 MMA: the `bdpas` block-scaled instruction only
 * accepts UE8M0 scale factors with a fixed block size of 32, while NVFP4 uses
 * E4M3 scale factors with block size 16 plus one FP32 per-tensor global scale.
 * NVFP4 weights are therefore emulated by dequantizing to BF16 and
 * requantizing to MXFP8, after which a normal MXFP8 block-scaled GEMM runs.
 *
 * nvfp4_woq_prep_xe35 rewrites only the *scales* into the pair the fused mainloop
 * consumes, leaving the 4-bit payload untouched in memory:
 *
 *   c    (E, N, K/16) bf16   per-16 multiplier
 *   sfmx (E, K/32, N) uint8  UE8M0 per-32 scale
 *
 * sfmx is MN-major because it feeds the Xe block-2D scale load unchanged; c is
 * row-major because the mainloop reads a lane's K/16 entries for one N as a
 * short contiguous vector.
 *
 * Numerics contract (see tests/nvfp4_utils.py for the reference):
 *
 *   s_j    = gs_dq * float(sf16_j)                     per-16 dequant scale
 *   amax32 = 6 * max(s_2j, s_2j+1)                     E2M1 magnitude bound
 *   kexp   = min k such that 448 * 2^k >= amax32       UE8M0 exponent
 *   c_j    = bf16(s_j * 2^-kexp)                       per-16 multiplier
 *   w_mx   = e4m3_rne(float(c_j) * float(e2m1_val))    MXFP8 value
 *
 * `6` is the largest E2M1 magnitude, so `amax32` upper-bounds the group and
 * requantization never clips.  NVFP4 quantizers pick sf16 = block_amax / 6,
 * which makes the bound tight in practice.  Deriving kexp from the scales alone
 * (rather than from the payload) is what lets this run without touching the
 * 4-bit data at all.
 */

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>
#include <sgl_kernel/xe35/nvfp4_woq.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "cutlass/float8.h"

namespace at::native::xpu {

namespace {

constexpr int UE8M0_BIAS = 127;
constexpr float E2M1_MAX = 6.0f;  // largest E2M1 magnitude; bounds a group

/// Smallest integer k with 448 * 2^k >= amax, exactly (no log2 round-off).
///
/// frexp gives amax = m * 2^e with m in [0.5, 1).  448 == 1.75 * 2^8, so
/// 448 * 2^k >= m * 2^e  <=>  1.75 >= m * 2^(e - 8 - k), and since m < 1 the
/// smallest admissible k is e - 9 when 2m <= 1.75 and e - 8 otherwise.
inline int ue8m0_exponent_xe35(float amax) {
  if (!(amax > 0.0f)) {
    return 0;
  }
  int e = 0;
  const float m = sycl::frexp(amax, &e);
  const int k = (m <= 0.875f) ? (e - 9) : (e - 8);
  return sycl::clamp(k, -UE8M0_BIAS, 127);
}

/// Construct 2^exponent exactly for the full UE8M0 exponent range.
inline float power_of_two_xe35(int exponent) {
  // IEEE-754 encodes 2^-127 as a subnormal; all remaining values use a biased
  // normal exponent. Callers pass only [-127, 127].
  const uint32_t bits = exponent == -127 ? 0x00400000u : static_cast<uint32_t>(exponent + 127) << 23;
  return sycl::bit_cast<float>(bits);
}

/// Convert a finite float to BF16 bits using round-to-nearest, ties-to-even.
inline uint16_t float_to_bf16_rne_bits(float value) {
  const uint32_t bits = sycl::bit_cast<uint32_t>(value);
  const uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

struct NvFp4WoqPrepKernel_xe35;

}  // namespace

// ---------------------------------------------------------------------------
// nvfp4_woq_prep_xe35
// ---------------------------------------------------------------------------
//
// One work-item per (expert, n, 32-group).  Scales only — the 4-bit payload is
// consumed straight from the checkpoint by the fused mainloop.  sfmx comes out
// MN-major so the mainloop can load it with the same Xe block-2D scale atoms
// the MXFP8 path already uses; c comes out row-major (N, K/16) so each lane's
// per-16 multipliers for one N are contiguous.
SGL_KERNEL_EXPORT void nvfp4_woq_prep_xe35(
    torch::Tensor& out_c, torch::Tensor& out_sfmx, const torch::Tensor& w_s, const torch::Tensor& w_gs) {
  TORCH_CHECK(w_s.device().is_xpu(), "nvfp4_woq_prep_xe35: w_s must be on XPU");
  const auto device = w_s.device();
  TORCH_CHECK(w_gs.device() == device, "nvfp4_woq_prep_xe35: w_gs must be on the same XPU device as w_s");
  TORCH_CHECK(out_c.device() == device, "nvfp4_woq_prep_xe35: out_c must be on the same XPU device as w_s");
  TORCH_CHECK(out_sfmx.device() == device, "nvfp4_woq_prep_xe35: out_sfmx must be on the same XPU device as w_s");
  TORCH_CHECK(w_s.dim() == 3, "nvfp4_woq_prep_xe35: w_s must be (E, N, K/16), got ", w_s.dim(), "D");
  TORCH_CHECK(w_s.scalar_type() == torch::kFloat8_e4m3fn, "nvfp4_woq_prep_xe35: w_s must be float8_e4m3fn");
  TORCH_CHECK(w_gs.scalar_type() == torch::kFloat32, "nvfp4_woq_prep_xe35: w_gs must be float32");
  TORCH_CHECK(out_c.scalar_type() == torch::kBFloat16, "nvfp4_woq_prep_xe35: out_c must be bfloat16");
  TORCH_CHECK(out_sfmx.scalar_type() == torch::kUInt8, "nvfp4_woq_prep_xe35: out_sfmx must be uint8 (UE8M0)");
  TORCH_CHECK(w_s.is_contiguous(), "nvfp4_woq_prep_xe35: w_s must be contiguous");
  TORCH_CHECK(w_gs.is_contiguous(), "nvfp4_woq_prep_xe35: w_gs must be contiguous");
  TORCH_CHECK(out_c.is_contiguous() && out_sfmx.is_contiguous(), "nvfp4_woq_prep_xe35: outputs must be contiguous");

  const int64_t E = w_s.size(0);
  const int64_t N = w_s.size(1);
  const int64_t sk16 = w_s.size(2);
  TORCH_CHECK(sk16 % 2 == 0, "nvfp4_woq_prep_xe35: K/16 must be even (K must be a multiple of 32), got ", sk16);
  const int64_t sk32 = sk16 / 2;

  TORCH_CHECK(w_gs.numel() == E, "nvfp4_woq_prep_xe35: w_gs must have E entries, got ", w_gs.numel());
  TORCH_CHECK(
      out_c.dim() == 3 && out_c.size(0) == E && out_c.size(1) == N && out_c.size(2) == sk16,
      "nvfp4_woq_prep_xe35: out_c must be (E, N, K/16)");
  TORCH_CHECK(
      out_sfmx.dim() == 3 && out_sfmx.size(0) == E && out_sfmx.size(1) == sk32 && out_sfmx.size(2) == N,
      "nvfp4_woq_prep_xe35: out_sfmx must be (E, K/32, N)");

  const int64_t total = E * N * sk32;
  if (total == 0) {
    return;
  }

  const uint8_t* src_s = reinterpret_cast<const uint8_t*>(w_s.data_ptr());
  const float* src_gs = w_gs.data_ptr<float>();
  auto* dst_c = reinterpret_cast<uint16_t*>(out_c.data_ptr());
  uint8_t* dst_sfmx = reinterpret_cast<uint8_t*>(out_sfmx.data_ptr());

  auto& q = at::xpu::getCurrentXPUStream(device.index()).queue();
  const size_t wg = 256;
  const size_t global = ((static_cast<size_t>(total) + wg - 1) / wg) * wg;

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<NvFp4WoqPrepKernel_xe35>(
        sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(wg)), [=](sycl::nd_item<1> item) {
          const int64_t gid = static_cast<int64_t>(item.get_global_linear_id());
          if (gid >= total) {
            return;
          }
          // n is the fastest-varying coordinate so the MN-major stores coalesce.
          const int64_t n = gid % N;
          const int64_t rest = gid / N;
          const int64_t g32 = rest % sk32;
          const int64_t e = rest / sk32;

          const int64_t sf_base = (e * N + n) * sk16 + 2 * g32;
          const float gs = src_gs[e];
          const auto* sf = reinterpret_cast<const cutlass::float_e4m3_t*>(src_s);
          const float s0 = static_cast<float>(sf[sf_base]) * gs;
          const float s1 = static_cast<float>(sf[sf_base + 1]) * gs;

          const int kexp = ue8m0_exponent_xe35(E2M1_MAX * sycl::fmax(s0, s1));
          const float inv = power_of_two_xe35(-kexp);

          dst_sfmx[(e * sk32 + g32) * N + n] = static_cast<uint8_t>(kexp + UE8M0_BIAS);
          auto* c_row = dst_c + (e * N + n) * sk16 + 2 * g32;
          c_row[0] = float_to_bf16_rne_bits(s0 * inv);
          c_row[1] = float_to_bf16_rne_bits(s1 * inv);
        });
  });
}

}  // namespace at::native::xpu
