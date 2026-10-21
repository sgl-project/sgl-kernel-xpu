/***************************************************************************************************
 * Copyright 2025 SGLang Team. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 **************************************************************************************************/
/*! \file
    \brief MXFP4 x MXFP4 (W4A4) MoE grouped GEMM for Xe3.5 (CRI), ported from
           vllm-xpu-kernels @ 48313ef (csrc/xpu/grouped_gemm/xe_3/).

    Uses vLLM's MainloopMXFPXGroup_xe35 plus its per-shape tile family and wave-quantization
    dispatch (see sycl/xe35/moe/collective/) instead of CUTLASS's in-tree
    MainloopIntelXeXMX16BlockScaledGroup with its single fixed 512x512x64 tile. On the
    larger expert groups, the array scheduler permits smaller tiles than the
    single fixed 512x512x64 tile in the in-tree blockwise collective.

    Layout contract (identical to vLLM's cutlass_grouped_gemm_interface, because the
    strides are rebuilt from {N, K, groups} and the tensors' own strides are never read):
      a         [total_M, K/2]        float4_e2m1fn_x2, row-major
      a_scale   [padded_M, K/32]      float8_e8m0fnu, M-dense, each expert's M rounded up to 4
      b         [E, N, K/2]           float4_e2m1fn_x2, contiguous (i.e. K-dense / ColumnMajor)
      b_scale   contiguous [E, K/32, N] float8_e8m0fnu, or its
                [E, N, K/32] transposed view; N-dense in storage
      out       [total_M, N]          bfloat16
      rows_per_expert [E]             int32, on device
*/

#include "sgl_kernel/xe35/mxfp4_w4a4.h"

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>


#include "sycl/xe35/moe/device/grouped_gemm_runner.hpp"

namespace gpu::cutlass_kernel::grouped_gemm_xe35 {

// A wider N tile reduces scheduler work for sufficiently populated expert
// groups in the decode regime.
class Mxfp4WideDecodePolicy_xe35 : public moe_mxfp4_policy_xe35 {
 public:
  using TileShape = Shape<_8, _256, _64>;
  using SGLayout = Layout<Shape<_1, _8, _1>, Stride<_8, _1, _0>>;
  using TiledMma =
      typename TiledMMAHelper<MMA_Atom<XE_BDPAS_TT<8, float, ElementA>>, Layout<TileShape>, SGLayout>::TiledMMA;
  static constexpr int PipelineStages = 2;
  using GEMMDispatchPolicy = cutlass::gemm::MainloopMXFPXGroup_xe35<PipelineStages>;
  CALL_GENERATE_GEMM();
};

}  // namespace gpu::cutlass_kernel::grouped_gemm_xe35

namespace {

using namespace sgl_kernel::vllm_grouped_gemm_xe35;
namespace policies = gpu::cutlass_kernel::grouped_gemm_xe35;

// The Xe 2D block load of the A-scales is DWord-granular, so the mainloop indexes each
// expert's block as ceil(K/32) * round_up(M_e, kScaleAlignRows) bytes.
constexpr int64_t kScaleAlignRows = 4;

void check_a_scale_rows_xe35(int64_t a_scale_rows, int64_t total_m) {
  TORCH_CHECK(
      a_scale_rows >= total_m,
      "a_scale must have at least as many rows as a (",
      total_m,
      "), since it holds sum(round_up(M_e, ",
      kScaleAlignRows,
      ")) >= sum(M_e) rows, got ",
      a_scale_rows);
}

// Selects between the two general prefill tiles.
bool use_256x256_prefill_tile_xe35(int64_t M_total, int64_t N, int64_t groups) {
  constexpr int kCores = 32;
  constexpr double kMinUtil = 0.85;
  const int64_t M_g = M_total / (groups > 0 ? groups : 1);
  auto ceil_div = [](int64_t a, int64_t b) { return (a + b - 1) / b; };
  int64_t tiles_256 = groups * ceil_div(M_g > 0 ? M_g : 1, 256) * ceil_div(N, 256);
  int64_t waves_256 = ceil_div(tiles_256, kCores);
  double util_256 = static_cast<double>(tiles_256) / static_cast<double>(waves_256 * kCores);

  return util_256 >= kMinUtil;
}

bool use_wide_decode_tile_xe35(int64_t avg_rows_per_expert, int64_t N) {
  // Device-event sweeps show two crossovers: mid wins at 2-8 rows, and
  // wide wins again at 9-16. At larger row counts, tail-wave utilization
  // depends on the number of N tiles. Favor mid where the wide tile's
  // p20-p80 variation overlaps or exceeds its median advantage.
  if (avg_rows_per_expert == 1) {
    return true;
  }
  if (avg_rows_per_expert <= 8) {
    return false;
  }
  if (avg_rows_per_expert <= 16) {
    return true;
  }
  const int64_t wide_n_tiles = (N + 255) / 256;
  if (wide_n_tiles <= 16) {
    return avg_rows_per_expert <= 18 || avg_rows_per_expert >= 25;
  }
  return avg_rows_per_expert >= 26;
}

}  // namespace

void moe_grouped_mm_nt_mxfp4_w4a4_xe35(
    torch::Tensor& output,
    const torch::Tensor& a,
    const torch::Tensor& a_scale,
    const torch::Tensor& b,
    const torch::Tensor& b_scale,
    const std::optional<at::Tensor>& bias,
    const torch::Tensor& rows_per_expert,
    const int64_t n,
    const int64_t k,
    const int64_t num_experts) {
  auto& dpcpp_queue = at::xpu::getCurrentXPUStream(a.device().index()).queue();

  const int64_t groups = num_experts;
  const int64_t N = n;
  const int64_t K = k;
  const int64_t total_m = a.size(0);
  const int64_t scale_k = (K + 31) / 32;

  TORCH_CHECK(groups > 0 && N > 0 && K > 0, "n, k and num_experts must be > 0");
  TORCH_CHECK(K % 64 == 0, "K must be a multiple of 64 (4-bit packing and the 32-element scale block)");

  // The XE 2D block copies need each row of B and D to start on a 128-bit boundary,
  // and cutlass reports a violation by aborting the process rather than returning.
  TORCH_CHECK(N % 32 == 0, "N must be a multiple of 32 so each 4-bit weight row is 128-bit aligned");

  TORCH_CHECK(
      a.device().is_xpu() && b.device().is_xpu() && a_scale.device().is_xpu() && b_scale.device().is_xpu() &&
          output.device().is_xpu() && rows_per_expert.device().is_xpu(),
      "all tensors must be on an XPU device");

  // Packed 4-bit data and e8m0 scales are also commonly carried as uint8 in
  // sgl-kernel-xpu, so accept either the byte view or the typed one.
  auto is_packed_fp4 = [](const torch::Tensor& t) {
    return t.scalar_type() == at::kFloat4_e2m1fn_x2 || t.scalar_type() == at::kByte;
  };
  auto is_e8m0 = [](const torch::Tensor& t) {
    return t.scalar_type() == at::kFloat8_e8m0fnu || t.scalar_type() == at::kByte;
  };
  TORCH_CHECK(
      is_packed_fp4(a) && is_packed_fp4(b),
      "a and b must be float4_e2m1fn_x2 (or its uint8 view); got ",
      a.scalar_type(),
      " and ",
      b.scalar_type());
  TORCH_CHECK(
      is_e8m0(a_scale) && is_e8m0(b_scale),
      "a_scale and b_scale must be float8_e8m0fnu (or its uint8 view); got ",
      a_scale.scalar_type(),
      " and ",
      b_scale.scalar_type());
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bfloat16");

  TORCH_CHECK(
      a.dim() == 2 && a.size(1) == K / 2 && a.is_contiguous(),
      "a must be a contiguous 2D [total_M, K/2] tensor of packed 4-bit activations");
  TORCH_CHECK(
      output.dim() == 2 && output.size(0) == total_m && output.size(1) == N && output.is_contiguous(),
      "output must be a contiguous 2D [total_M, N] tensor");
  TORCH_CHECK(
      b.dim() == 3 && b.size(0) == groups && b.size(1) == N && b.size(2) == K / 2 && b.is_contiguous(),
      "b must be a contiguous [num_experts, N, K/2] tensor");
  TORCH_CHECK(
      rows_per_expert.dim() == 1 && rows_per_expert.numel() == groups && rows_per_expert.scalar_type() == at::kInt,
      "rows_per_expert must be a 1D int32 tensor with one entry per expert");
  TORCH_CHECK(rows_per_expert.is_contiguous(), "rows_per_expert must be contiguous");

  // Only the extent of the A-scale surface is meaningful here: it is a plain
  // [padded_M, scale_k] buffer filled MN-major with each expert's row count rounded
  // up to a multiple of 4.
  TORCH_CHECK(
      a_scale.dim() == 2 && a_scale.size(1) == scale_k && a_scale.is_contiguous(),
      "a_scale must be a contiguous [padded_M, ceil(K/32)] tensor");
  // Callers that pad each expert's row block size the activation buffer from a host-side
  // upper bound rather than syncing the routing histogram, so a_scale is allowed to be
  // taller than the rows actually routed.
  check_a_scale_rows_xe35(a_scale.size(0), total_m);
  // Element (e, n, k) must sit at e * N * scale_k + n + k * N.
  TORCH_CHECK(
      b_scale.dim() == 3 && b_scale.size(0) == groups &&
          ((b_scale.size(1) == scale_k && b_scale.size(2) == N && b_scale.is_contiguous()) ||
           (b_scale.size(1) == N && b_scale.size(2) == scale_k && b_scale.stride(0) == N * scale_k &&
            (N == 1 || b_scale.stride(1) == 1) && (scale_k == 1 || b_scale.stride(2) == N))),
      "b_scale must be dense along N: either a contiguous [num_experts, ceil(K/32), N] tensor "
      "or its [num_experts, N, ceil(K/32)] transposed view");

  // The epilogue reads C as one fp32 row per token, so a per-expert bias must already
  // be expanded by the caller.
  if (bias) {
    TORCH_CHECK(
        bias->dim() == 2 && bias->size(0) == total_m && bias->size(1) == N && bias->is_contiguous() &&
            bias->scalar_type() == at::kFloat,
        "bias must be a contiguous fp32 [total_M, N] tensor");
  }

  const int64_t avg_tokens_cnt = total_m / groups;

#define SGL_GG_CALL(POLICY)              \
  kernel_functor_xe35<POLICY>(                \
      dpcpp_queue,                       \
      a.data_ptr(),                      \
      a_scale.data_ptr(),                \
      b.data_ptr(),                      \
      b_scale.data_ptr(),                \
      bias ? bias->data_ptr() : nullptr, \
      output.data_ptr(),                 \
      rows_per_expert.data_ptr(),        \
      N,                                 \
      K,                                 \
      groups)

  if (avg_tokens_cnt > 32) {
    if (K <= 1024 && N >= 1024) {
      // Narrow-K down-projection: a wider N tile amortizes the short K loop.
      if (avg_tokens_cnt <= 2048) {
        SGL_GG_CALL(policies::moe_mxfp4_256x128_policy_xe35);
      } else {
        SGL_GG_CALL(policies::moe_mxfp4_downproj_wide_policy_xe35);
      }
    } else {
      if (use_256x256_prefill_tile_xe35(total_m, N, groups)) {
        SGL_GG_CALL(policies::moe_mxfp4_policy_xe35);
      } else {
        SGL_GG_CALL(policies::moe_mxfp4_128x256_policy_xe35);
      }
    }
  } else if (groups >= 32 && K >= 2048 && N >= 4096 && avg_tokens_cnt > 0) {
    if (use_wide_decode_tile_xe35(avg_tokens_cnt, N)) {
      SGL_GG_CALL(policies::Mxfp4WideDecodePolicy_xe35);
    } else {
      SGL_GG_CALL(policies::moe_mxfp4_mid_policy_xe35);
    }
  } else if (avg_tokens_cnt > 4) {
    SGL_GG_CALL(policies::moe_mxfp4_mid_policy_xe35);
  } else {
    SGL_GG_CALL(policies::moe_mxfp4_decode_policy_xe35);
  }

#undef SGL_GG_CALL
}
