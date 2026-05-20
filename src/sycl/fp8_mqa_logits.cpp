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
// Implements fp8_mqa_logits (ragged) and fp8_paged_mqa_logits (paged),
// matching DeepGEMM semantics on CUDA.

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include "kernels/nsa/fp8_mqa_logits_kernel.hpp"

namespace {

constexpr int WG_SIZE = 256;

// Round up division
inline int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

}  // namespace

// fp8_mqa_logits: prefill/extend path
// q_fp8: (Nq, H, D) uint8 (fp8 e4m3)
// k_fp8: (Nk, D) uint8 (fp8 e4m3)
// k_scale: (Nk,) float32
// weights: (Nq, H) float32
// ks: (Nq,) int32
// ke: (Nq,) int32
// Returns: logits (Nq, Nk) float32
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
  TORCH_CHECK(k_scale.dim() == 1, "k_scale must be 1D (Nk,)");
  TORCH_CHECK(weights.dim() == 2, "weights must be 2D (Nq, H)");

  int Nq = q_fp8.size(0);
  int H = q_fp8.size(1);
  int D = q_fp8.size(2);
  int Nk = k_fp8.size(0);

  TORCH_CHECK(k_fp8.size(1) == D, "k_fp8 dim mismatch");
  TORCH_CHECK(k_scale.size(0) == Nk, "k_scale size mismatch");
  TORCH_CHECK(weights.size(0) == Nq && weights.size(1) == H, "weights shape mismatch");
  TORCH_CHECK(ks.size(0) == Nq && ke.size(0) == Nq, "ks/ke size mismatch");

  auto logits = torch::zeros({Nq, Nk}, torch::dtype(torch::kFloat32).device(q_fp8.device()));

  if (Nq == 0 || Nk == 0) return logits;

  auto stream = c10::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  nsa::Fp8MqaLogitsKernel<> kernel{
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

  // Grid: (Nq, Nk) with work-group size along dim1
  int tile_k = 256;
  sycl::range<2> global(Nq, cdiv(Nk, 1) * 1);
  // Use 1 work-item per (qi, kj) pair for the naive kernel
  sycl::range<2> global_range(Nq, Nk);
  sycl::range<2> local_range(1, std::min(Nk, WG_SIZE));

  // Adjust global range to be a multiple of local range
  int global_k = cdiv(Nk, (int)local_range[1]) * local_range[1];
  sycl::range<2> adjusted_global(Nq, global_k);

  queue.submit([&](sycl::handler& cgh) { cgh.parallel_for(sycl::nd_range<2>(adjusted_global, local_range), kernel); });

  return logits;
}

// fp8_paged_mqa_logits: decode path
// q_fp8: (B, 1, H, D) uint8 (fp8 e4m3) — B is batch * next_n
// kv_cache: (num_pages, page_size, 1, D+4) uint8
// weights: (B, H) float32
// seq_lens: (B,) or (B,1) int32
// block_tables: (B, max_num_blocks) int32
// max_seq_len: int
// Returns: logits (B, max_seq_len) float32
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
  TORCH_CHECK(kv_cache.dim() == 4, "kv_cache must be 4D (num_pages, page_size, 1, D+4)");

  int B_next = q_fp8.size(0);
  // q_fp8 shape: (B, next_n=1, H, D)
  int H = q_fp8.size(2);
  int D = q_fp8.size(3);
  int page_size = kv_cache.size(1);
  int head_dim_with_sf = kv_cache.size(3);
  int max_num_blocks = block_tables.size(1);

  TORCH_CHECK(head_dim_with_sf == D + 4, "KV cache last dim must be D+4");
  TORCH_CHECK(weights.size(0) == B_next && weights.size(1) == H, "weights shape mismatch");

  auto logits = torch::zeros({B_next, max_seq_len}, torch::dtype(torch::kFloat32).device(q_fp8.device()));

  if (B_next == 0 || max_seq_len == 0) return logits;

  // Flatten seq_lens to 1D
  auto seq_lens_flat = seq_lens.dim() == 2 ? seq_lens.contiguous().view({-1}) : seq_lens.contiguous();

  auto stream = c10::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

  // q is (B, 1, H, D); skip the next_n=1 dimension for pointer arithmetic
  nsa::Fp8PagedMqaLogitsKernel kernel{
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
      static_cast<int>(max_seq_len),
      page_size * head_dim_with_sf  // kv_stride per page
  };

  sycl::range<2> local_range(1, std::min(static_cast<int>(max_seq_len), WG_SIZE));
  int global_k = cdiv(static_cast<int>(max_seq_len), (int)local_range[1]) * local_range[1];
  sycl::range<2> adjusted_global(B_next, global_k);

  queue.submit([&](sycl::handler& cgh) { cgh.parallel_for(sycl::nd_range<2>(adjusted_global, local_range), kernel); });

  return logits;
}
