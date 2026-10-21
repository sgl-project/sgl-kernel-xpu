#pragma once

#include <torch/all.h>

#pragma GCC visibility push(default)

// MXFP4-A x MXFP4-B (W4A4) MoE grouped GEMM using the vLLM Xe3 collective
// (src/sycl/xe35/moe/), which carries a per-shape tile family and
// wave-quantization dispatch instead of a single fixed tile. Takes the problem shape as
// scalars plus a device-side rows_per_expert, so no ptr tables or host-visible
// problem_sizes are needed.
//   a [total_M, K/2], a_scale [padded_M, K/32] (each expert's M padded up to 4),
//   b [E, N, K/2], b_scale [E, N, K/32] (N-dense), output [total_M, N] bf16.
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
    const int64_t num_experts);
void mxfp4_moe_scale_to_mn_major_xe35(torch::Tensor& out, const torch::Tensor& in, const torch::Tensor& rows_per_expert);

#pragma GCC visibility pop
