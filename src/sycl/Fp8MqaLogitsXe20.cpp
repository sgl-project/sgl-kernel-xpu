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

// FP8 paged MQA Logits kernel for NSA (Native Sparse Attention) indexer scoring (decode path).
// Optimized path: SYCL-TLA FP8 GEMM (fused FP8→bf16 conversion + XMX GEMM)
// + lightweight reduction kernel for ReLU + weighted head sum + scaling.
// Naive SYCL fallback for small problem sizes.
// Note: fp8_mqa_logits (prefill path) is implemented in pure Python via sgl_kernel.nsa.

#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include "kernels/nsa/fp8_mqa_gemm_xe20.hpp"
#include "kernels/nsa/fp8_mqa_logits_kernel.hpp"

// Thin launcher for the custom FP8 MQA GEMM on Intel BMG (Xe20).
// The kernel implementation is in kernels/nsa/fp8_mqa_gemm_xe20.hpp.
// Batched variant: computes `batch` independent GEMMs D_b = A_b(M,K) @ B_b(N,K)^T
// in a single fused launch. Per-batch base pointers advance by the given strides
// (in elements). All batches share the same (M,N,K) tile shape. B is in (N,K)
// layout with K contiguous — no host-side transpose needed.
// Returns 0 on success, non-zero if dimensions are not tile-aligned.
int fp8_mqa_gemm_xe20_batched(
    sycl::queue* queue_ptr,
    const void* A_fp8,
    const void* B_fp8,
    void* D_f32,
    int batch,
    int M,
    int N,
    int K,
    int64_t A_batch_stride,
    int64_t B_batch_stride,
    int64_t D_batch_stride) {
  return nsa::fp8_mqa_gemm_batched_launch(
      queue_ptr, A_fp8, B_fp8, D_f32, batch, M, N, K, A_batch_stride, B_batch_stride, D_batch_stride);
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

// Batched FP8 GEMM via SYCL-TLA: for each b, D_b(M,N) = A_b(M,K) @ B_b(N,K)^T.
// Base pointers advance by the per-batch element strides. Writes directly into
// the pre-allocated output buffer. Falls back to torch::bmm if not tile-aligned.
int fp8_gemm_xe20_batched_inplace(
    sycl::queue& queue,
    const uint8_t* a_ptr,  // (batch, M, K) fp8
    const uint8_t* b_ptr,  // (batch, N, K) fp8
    float* d_ptr,          // (batch, M, N) output
    int batch,
    int M,
    int N,
    int K,
    int64_t a_batch_stride,
    int64_t b_batch_stride,
    int64_t d_batch_stride,
    at::Device device) {
  int rc = fp8_mqa_gemm_xe20_batched(
      &queue, a_ptr, b_ptr, d_ptr, batch, M, N, K, a_batch_stride, b_batch_stride, d_batch_stride);

  if (rc != 0) {
    // Fallback to torch::bmm if dimensions are not tile-aligned.
    // Strides are the natural contiguous strides (M*K, N*K, M*N).
    auto a_fp8 = torch::from_blob(const_cast<uint8_t*>(a_ptr), {batch, M, K}, torch::dtype(torch::kByte).device(device))
                     .view(at::ScalarType::Float8_e4m3fn);
    auto b_fp8 = torch::from_blob(const_cast<uint8_t*>(b_ptr), {batch, N, K}, torch::dtype(torch::kByte).device(device))
                     .view(at::ScalarType::Float8_e4m3fn);
    auto a_bf16 = a_fp8.to(at::ScalarType::BFloat16);
    auto b_bf16 = b_fp8.to(at::ScalarType::BFloat16);
    auto result = at::bmm(a_bf16, b_bf16.transpose(1, 2)).to(at::ScalarType::Float);
    auto d_view = torch::from_blob(d_ptr, {batch, M, N}, torch::dtype(torch::kFloat32).device(device));
    d_view.copy_(result);
  }
  return rc;
}

}  // namespace

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
  TORCH_CHECK(
      !schedule_metadata.has_value(),
      "fp8_paged_mqa_logits does not support schedule_metadata on XPU");
  TORCH_CHECK(q_fp8.is_xpu(), "q_fp8 must be on XPU");
  TORCH_CHECK(kv_cache.is_xpu(), "kv_cache must be on XPU");
  TORCH_CHECK(q_fp8.dim() == 4, "q_fp8 must be 4D (B, 1, H, D)");
  TORCH_CHECK(kv_cache.dim() == 4, "kv_cache must be 4D");
  TORCH_CHECK(q_fp8.size(1) == 1, "q_fp8 must have shape (B, 1, H, D) with size(1)=1");
  TORCH_CHECK(q_fp8.scalar_type() == at::kByte, "q_fp8 must be uint8 (FP8 e4m3)");
  TORCH_CHECK(kv_cache.scalar_type() == at::kByte, "kv_cache must be uint8");
  TORCH_CHECK(weights.scalar_type() == at::kFloat, "weights must be float32");
  TORCH_CHECK(seq_lens.scalar_type() == at::kInt, "seq_lens must be int32");
  TORCH_CHECK(block_tables.scalar_type() == at::kInt, "block_tables must be int32");
  TORCH_CHECK(block_tables.dim() == 2, "block_tables must be 2D (B, max_num_blocks)");
  TORCH_CHECK(block_tables.size(0) == q_fp8.size(0), "block_tables batch size must match q_fp8 batch size");
  // schedule_metadata is accepted for API compatibility with DeepGEMM but not
  // used on XPU — scheduling is handled internally by the SYCL runtime.
  if (schedule_metadata.has_value()) {
    TORCH_WARN_ONCE("fp8_paged_mqa_logits: schedule_metadata is ignored on XPU");
  }

  int B_next = q_fp8.size(0);
  int H = q_fp8.size(2);
  int D = q_fp8.size(3);
  int page_size = kv_cache.size(1);
  int head_dim_with_sf = kv_cache.size(3);
  int max_num_blocks = block_tables.size(1);
  int msl = static_cast<int>(max_seq_len);

  TORCH_CHECK(
      head_dim_with_sf == D + 4,
      "kv_cache last dim must be D+4 (FP8 key data + float32 scale), got ",
      head_dim_with_sf,
      " vs D+4=",
      D + 4);
  TORCH_CHECK(D % 4 == 0, "D must be a multiple of 4 for vectorized gather, got ", D);
  TORCH_CHECK(
      static_cast<int64_t>(max_num_blocks) * page_size >= msl,
      "block_tables capacity (",
      max_num_blocks * page_size,
      ") must cover max_seq_len (",
      msl,
      ")");
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == B_next && weights.size(1) == H, "weights must be (B, H)");

  // clean_logits is accepted for API compatibility but output is always
  // zero-initialized — the cost is negligible relative to GEMM.
  (void)clean_logits;
  auto logits = torch::zeros({B_next, msl}, torch::dtype(torch::kFloat32).device(q_fp8.device()));
  if (B_next == 0 || msl == 0) return logits;

  auto seq_lens_flat = seq_lens.dim() == 2 ? seq_lens.contiguous().view({-1}) : seq_lens.contiguous();
  TORCH_CHECK(
      seq_lens_flat.size(0) == B_next,
      "seq_lens must have B elements after flattening, got ",
      seq_lens_flat.size(0),
      " vs B=",
      B_next);

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

    // Stage 2: Batched SYCL-TLA FP8 GEMM — all B_next GEMMs in one launch.
    // TODO: switch to torch._scaled_grouped_mm once it is implemented on XPU
    // (currently aten::_scaled_grouped_mm_v2 is unimplemented and 3D batched
    // _scaled_mm is unsupported, so the only torch option is a per-batch
    // _scaled_mm loop that does not scale with B). The custom batched SYCL-TLA
    // kernel is 4-15x faster than that loop for B>=4, so we keep it for decode.
    auto dots = torch::empty({B_next, H, msl}, torch::dtype(torch::kFloat32).device(q_fp8.device()));

    fp8_gemm_xe20_batched_inplace(
        queue,
        q_contig.data_ptr<uint8_t>(),    // (B,1,H,D), A batch stride = H*D
        k_gathered.data_ptr<uint8_t>(),  // (B*msl,D), B batch stride = msl*D
        dots.data_ptr<float>(),          // (B,H,msl), D batch stride = H*msl
        B_next,
        H,
        msl,
        D,
        static_cast<int64_t>(H) * D,
        static_cast<int64_t>(msl) * D,
        static_cast<int64_t>(H) * msl,
        q_fp8.device());

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
