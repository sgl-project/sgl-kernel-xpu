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

// FP8 MQA Logits kernels for NSA (Native Sparse Attention) indexer scoring.
// Optimized path: SYCL-TLA FP8 GEMM (fused FP8→bf16 conversion + XMX GEMM)
// + lightweight reduction kernel for ReLU + weighted head sum + scaling.
// Naive SYCL fallback for small problem sizes.

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include "kernels/nsa/fp8_mqa_logits_kernel.hpp"

// SYCL-TLA FP8 GEMM: D(M,N) = A_fp8(M,K) @ B_fp8(K,N) in f32.
// B must be in (K,N) row-major layout (N contiguous).
extern int
fp8_mqa_gemm_xe20(sycl::queue* queue_ptr, const void* A_fp8, const void* B_fp8, void* D_f32, int M, int N, int K);

namespace {

constexpr int WG_SIZE = 256;
// SYCL-TLA tile: 32x128x32. M must be ≥32, N must be ≥128.
constexpr int MIN_M_GEMM = 32;
constexpr int MIN_N_GEMM = 128;

inline int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

template <typename Kernel>
void launch_2d_kernel(sycl::queue& queue, Kernel& kernel, int dim0, int dim1) {
  sycl::range<2> local_range(1, std::min(dim1, WG_SIZE));
  int global_1 = cdiv(dim1, (int)local_range[1]) * local_range[1];
  sycl::range<2> adjusted_global(dim0, global_1);
  queue.submit([&](sycl::handler& cgh) { cgh.parallel_for(sycl::nd_range<2>(adjusted_global, local_range), kernel); });
}

// FP8 GEMM via SYCL-TLA: D(M,N) = A(M,K) @ B(N,K)^T in f32.
// Handles transposing B from (N,K) to (K,N) for CUTLASS.
at::Tensor fp8_gemm_xe20(
    sycl::queue& queue,
    const at::Tensor& a_uint8,  // (M, K) uint8 containing fp8 e4m3
    const at::Tensor& b_uint8,  // (N, K) uint8 containing fp8 e4m3
    int M,
    int N,
    int K) {
  // CUTLASS expects B in (K,N) row-major. Input is (N,K) row-major → transpose.
  auto b_transposed = b_uint8.view({N, K}).t().contiguous();  // (K, N)

  auto D = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(a_uint8.device()));

  int rc = fp8_mqa_gemm_xe20(
      &queue, a_uint8.data_ptr<uint8_t>(), b_transposed.data_ptr<uint8_t>(), D.data_ptr<float>(), M, N, K);

  if (rc != 0) {
    // Fallback to torch::mm if SYCL-TLA can_implement fails (e.g. size not aligned)
    auto a_fp8 = a_uint8.view({M, K}).view(at::ScalarType::Float8_e4m3fn);
    auto b_fp8 = b_uint8.view({N, K}).view(at::ScalarType::Float8_e4m3fn);
    auto a_bf16 = a_fp8.to(at::ScalarType::BFloat16);
    auto b_bf16 = b_fp8.to(at::ScalarType::BFloat16);
    return at::mm(a_bf16, b_bf16.t()).to(at::ScalarType::Float);
  }
  return D;
}

}  // namespace

// fp8_mqa_logits: prefill/extend path
torch::Tensor fp8_mqa_logits(
    const torch::Tensor& q_fp8,
    const torch::Tensor& k_fp8,
    const torch::Tensor& k_scale,
    const torch::Tensor& weights,
    const torch::Tensor& ks,
    const torch::Tensor& ke) {
  TORCH_CHECK(q_fp8.is_xpu(), "q_fp8 must be on XPU");
  TORCH_CHECK(k_fp8.is_xpu(), "k_fp8 must be on XPU");
  TORCH_CHECK(q_fp8.dim() == 3, "q_fp8 must be 3D (Nq, H, D)");
  TORCH_CHECK(k_fp8.dim() == 2, "k_fp8 must be 2D (Nk, D)");

  int Nq = q_fp8.size(0);
  int H = q_fp8.size(1);
  int D = q_fp8.size(2);
  int Nk = k_fp8.size(0);

  auto logits = torch::zeros({Nq, Nk}, torch::dtype(torch::kFloat32).device(q_fp8.device()));
  if (Nq == 0 || Nk == 0) return logits;

  auto stream = c10::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  int M = Nq * H;
  bool use_gemm = (M >= MIN_M_GEMM && Nk >= MIN_N_GEMM);

  if (use_gemm) {
    // Optimized path: SYCL-TLA FP8 GEMM + reduction kernel
    auto dots = fp8_gemm_xe20(queue, q_fp8.contiguous().view({M, D}), k_fp8.contiguous(), M, Nk, D);

    nsa::Fp8MqaLogitsReduceKernel reduce_kernel{
        dots.data_ptr<float>(),
        weights.data_ptr<float>(),
        k_scale.data_ptr<float>(),
        ks.data_ptr<int32_t>(),
        ke.data_ptr<int32_t>(),
        logits.data_ptr<float>(),
        Nq,
        H,
        Nk};
    launch_2d_kernel(queue, reduce_kernel, Nq, Nk);
    return logits;
  }

  // Naive kernel fallback
  nsa::Fp8MqaLogitsKernel<32> kernel{
      q_fp8.data_ptr<uint8_t>(),
      k_fp8.data_ptr<uint8_t>(),
      k_scale.data_ptr<float>(),
      weights.data_ptr<float>(),
      ks.data_ptr<int32_t>(),
      ke.data_ptr<int32_t>(),
      logits.data_ptr<float>(),
      Nq,
      Nk,
      H,
      D};
  launch_2d_kernel(queue, kernel, Nq, Nk);
  return logits;
}

// fp8_paged_mqa_logits: decode path
torch::Tensor fp8_paged_mqa_logits(
    const torch::Tensor& q_fp8,
    const torch::Tensor& kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& seq_lens,
    const torch::Tensor& block_tables,
    const std::optional<torch::Tensor>& schedule_metadata,
    int64_t max_seq_len,
    bool clean_logits) {
  TORCH_CHECK(q_fp8.is_xpu(), "q_fp8 must be on XPU");
  TORCH_CHECK(kv_cache.is_xpu(), "kv_cache must be on XPU");
  TORCH_CHECK(q_fp8.dim() == 4, "q_fp8 must be 4D (B, 1, H, D)");
  TORCH_CHECK(kv_cache.dim() == 4, "kv_cache must be 4D");

  int B_next = q_fp8.size(0);
  int H = q_fp8.size(2);
  int D = q_fp8.size(3);
  int page_size = kv_cache.size(1);
  int head_dim_with_sf = kv_cache.size(3);
  int max_num_blocks = block_tables.size(1);
  int msl = static_cast<int>(max_seq_len);

  auto logits = torch::zeros({B_next, msl}, torch::dtype(torch::kFloat32).device(q_fp8.device()));
  if (B_next == 0 || msl == 0) return logits;

  auto seq_lens_flat = seq_lens.dim() == 2 ? seq_lens.contiguous().view({-1}) : seq_lens.contiguous();

  auto stream = c10::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  bool use_gemm = (H >= MIN_M_GEMM && msl >= MIN_N_GEMM);

  if (use_gemm) {
    // Stage 1: Gather K from pages into contiguous buffer
    auto k_gathered = torch::empty({B_next * msl, D}, torch::dtype(torch::kUInt8).device(q_fp8.device()));
    auto k_scale_gathered = torch::zeros({B_next, msl}, torch::dtype(torch::kFloat32).device(q_fp8.device()));

    nsa::PagedKGatherKernel gather_kernel{
        kv_cache.data_ptr<uint8_t>(),
        block_tables.data_ptr<int32_t>(),
        seq_lens_flat.data_ptr<int32_t>(),
        k_gathered.data_ptr<uint8_t>(),
        k_scale_gathered.data_ptr<float>(),
        B_next,
        D,
        page_size,
        max_num_blocks,
        msl,
        head_dim_with_sf};
    launch_2d_kernel(queue, gather_kernel, B_next, msl);

    // Stage 2: Per-batch SYCL-TLA FP8 GEMM
    auto dots = torch::empty({B_next, H, msl}, torch::dtype(torch::kFloat32).device(q_fp8.device()));

    for (int b = 0; b < B_next; ++b) {
      auto q_b = q_fp8.slice(0, b, b + 1).view({H, D});         // (H, D)
      auto k_b = k_gathered.slice(0, b * msl, (b + 1) * msl);   // (msl, D)
      auto dots_b = fp8_gemm_xe20(queue, q_b, k_b, H, msl, D);  // (H, msl)
      dots.slice(0, b, b + 1).view({H, msl}).copy_(dots_b);
    }

    // Stage 3: Reduction
    nsa::Fp8PagedMqaLogitsReduceKernel reduce_kernel{
        dots.data_ptr<float>(),
        weights.data_ptr<float>(),
        k_scale_gathered.data_ptr<float>(),
        seq_lens_flat.data_ptr<int32_t>(),
        logits.data_ptr<float>(),
        B_next,
        H,
        msl};
    launch_2d_kernel(queue, reduce_kernel, B_next, msl);
    return logits;
  }

  // Naive kernel fallback
  int kv_stride = page_size * 1 * head_dim_with_sf;
  nsa::Fp8PagedMqaLogitsKernel paged_kernel{
      q_fp8.data_ptr<uint8_t>(),
      kv_cache.data_ptr<uint8_t>(),
      weights.data_ptr<float>(),
      seq_lens_flat.data_ptr<int32_t>(),
      block_tables.data_ptr<int32_t>(),
      logits.data_ptr<float>(),
      B_next,
      H,
      D,
      page_size,
      max_num_blocks,
      msl,
      kv_stride};
  launch_2d_kernel(queue, paged_kernel, B_next, msl);
  return logits;
}
