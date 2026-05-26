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

#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include "kernels/nsa/fp8_mqa_gemm_xe20.hpp"
#include "kernels/nsa/fp8_mqa_logits_kernel.hpp"

// Thin launcher for the custom FP8 MQA GEMM on Intel BMG (Xe20).
// The kernel implementation is in kernels/nsa/fp8_mqa_gemm_xe20.hpp.
// D(M,N) = A_fp8(M,K) @ B_fp8(N,K)^T, all uint8 fp8 inputs, float32 output.
// B is in (N,K) layout with K contiguous — no host-side transpose needed.
// Returns 0 on success, non-zero if dimensions are not tile-aligned.
int fp8_mqa_gemm_xe20(sycl::queue* queue_ptr, const void* A_fp8, const void* B_fp8, void* D_f32, int M, int N, int K) {
  return nsa::fp8_mqa_gemm_launch(queue_ptr, A_fp8, B_fp8, D_f32, M, N, K);
}

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
// B is passed directly in (N,K) layout — no host-side transpose needed.
at::Tensor fp8_gemm_xe20(
    sycl::queue& queue,
    const at::Tensor& a_uint8,  // (M, K) uint8 containing fp8 e4m3
    const at::Tensor& b_uint8,  // (N, K) uint8 containing fp8 e4m3
    int M,
    int N,
    int K) {
  auto D = torch::zeros({M, N}, torch::dtype(torch::kFloat32).device(a_uint8.device()));

  int rc =
      fp8_mqa_gemm_xe20(&queue, a_uint8.data_ptr<uint8_t>(), b_uint8.data_ptr<uint8_t>(), D.data_ptr<float>(), M, N, K);

  if (rc != 0) {
    // Fallback to torch::mm if dimensions are not tile-aligned
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
  TORCH_CHECK(q_fp8.scalar_type() == at::kByte, "q_fp8 must be uint8 (FP8 e4m3)");
  TORCH_CHECK(k_fp8.scalar_type() == at::kByte, "k_fp8 must be uint8 (FP8 e4m3)");

  int Nq = q_fp8.size(0);
  int H = q_fp8.size(1);
  int D = q_fp8.size(2);
  int Nk = k_fp8.size(0);

  auto logits = torch::zeros({Nq, Nk}, torch::dtype(torch::kFloat32).device(q_fp8.device()));
  if (Nq == 0 || Nk == 0) return logits;

  auto stream = c10::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  // Ensure contiguity for all inputs passed to kernels via data_ptr
  auto q_contig = q_fp8.contiguous();
  auto k_contig = k_fp8.contiguous();
  auto k_scale_contig = k_scale.contiguous();
  auto weights_contig = weights.contiguous();
  auto ks_contig = ks.contiguous();
  auto ke_contig = ke.contiguous();

  int M = Nq * H;
  bool use_gemm = (M >= MIN_M_GEMM && Nk >= MIN_N_GEMM);

  if (use_gemm) {
    // Optimized path: SYCL-TLA FP8 GEMM + reduction kernel
    auto dots = fp8_gemm_xe20(queue, q_contig.view({M, D}), k_contig, M, Nk, D);

    nsa::Fp8MqaLogitsReduceKernel reduce_kernel{
        dots.data_ptr<float>(),
        weights_contig.data_ptr<float>(),
        k_scale_contig.data_ptr<float>(),
        ks_contig.data_ptr<int32_t>(),
        ke_contig.data_ptr<int32_t>(),
        logits.data_ptr<float>(),
        Nq,
        H,
        Nk};
    launch_2d_kernel(queue, reduce_kernel, Nq, Nk);
    return logits;
  }

  // Naive kernel fallback
  nsa::Fp8MqaLogitsKernel kernel{
      q_contig.data_ptr<uint8_t>(),
      k_contig.data_ptr<uint8_t>(),
      k_scale_contig.data_ptr<float>(),
      weights_contig.data_ptr<float>(),
      ks_contig.data_ptr<int32_t>(),
      ke_contig.data_ptr<int32_t>(),
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
  TORCH_CHECK(q_fp8.scalar_type() == at::kByte, "q_fp8 must be uint8 (FP8 e4m3)");
  TORCH_CHECK(kv_cache.scalar_type() == at::kByte, "kv_cache must be uint8");

  int B_next = q_fp8.size(0);
  int H = q_fp8.size(2);
  int D = q_fp8.size(3);
  int page_size = kv_cache.size(1);
  int head_dim_with_sf = kv_cache.size(3);
  int max_num_blocks = block_tables.size(1);
  int msl = static_cast<int>(max_seq_len);

  // Use torch::zeros when clean_logits is requested (out-of-range positions = 0),
  // torch::empty otherwise (out-of-range positions are undefined).
  auto logits = clean_logits ? torch::zeros({B_next, msl}, torch::dtype(torch::kFloat32).device(q_fp8.device()))
                             : torch::empty({B_next, msl}, torch::dtype(torch::kFloat32).device(q_fp8.device()));
  if (B_next == 0 || msl == 0) return logits;

  auto seq_lens_flat = seq_lens.dim() == 2 ? seq_lens.contiguous().view({-1}) : seq_lens.contiguous();

  // Ensure contiguity
  auto q_contig = q_fp8.contiguous();
  auto kv_contig = kv_cache.contiguous();
  auto weights_contig = weights.contiguous();
  auto block_tables_contig = block_tables.contiguous();

  auto stream = c10::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  bool use_gemm = (H >= MIN_M_GEMM && msl >= MIN_N_GEMM);

  if (use_gemm) {
    // Stage 1: Gather K from pages into contiguous buffer
    auto k_gathered = torch::empty({B_next * msl, D}, torch::dtype(torch::kUInt8).device(q_fp8.device()));
    auto k_scale_gathered = torch::zeros({B_next, msl}, torch::dtype(torch::kFloat32).device(q_fp8.device()));

    nsa::PagedKGatherKernel gather_kernel{
        kv_contig.data_ptr<uint8_t>(),
        block_tables_contig.data_ptr<int32_t>(),
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
    // TODO: batch all B_next GEMMs into a single grouped GEMM launch
    auto dots = torch::empty({B_next, H, msl}, torch::dtype(torch::kFloat32).device(q_fp8.device()));

    for (int b = 0; b < B_next; ++b) {
      auto q_b = q_contig.slice(0, b, b + 1).view({H, D});      // (H, D)
      auto k_b = k_gathered.slice(0, b * msl, (b + 1) * msl);   // (msl, D)
      auto dots_b = fp8_gemm_xe20(queue, q_b, k_b, H, msl, D);  // (H, msl)
      dots.slice(0, b, b + 1).view({H, msl}).copy_(dots_b);
    }

    // Stage 3: Reduction
    nsa::Fp8PagedMqaLogitsReduceKernel reduce_kernel{
        dots.data_ptr<float>(),
        weights_contig.data_ptr<float>(),
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
  nsa::Fp8PagedMqaLogitsKernel paged_kernel{
      q_contig.data_ptr<uint8_t>(),
      kv_contig.data_ptr<uint8_t>(),
      weights_contig.data_ptr<float>(),
      seq_lens_flat.data_ptr<int32_t>(),
      block_tables_contig.data_ptr<int32_t>(),
      logits.data_ptr<float>(),
      B_next,
      H,
      D,
      page_size,
      max_num_blocks,
      msl};
  launch_2d_kernel(queue, paged_kernel, B_next, msl);
  return logits;
}

#undef SYCL_INTEL_TARGET
