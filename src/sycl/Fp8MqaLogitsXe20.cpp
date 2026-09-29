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

#include <cstdlib>
#include <utility>

#include "SGLKernelPerf.h"
#include "SYCLHelpers.h"
#include "Utils.h"
#include "kernels/nsa/fp8_mqa_gemm_xe20.hpp"
#include "kernels/nsa/fp8_mqa_logits_kernel.hpp"
#include "sgl_kernel_export.h"

namespace {

constexpr int WG_SIZE = 256;
// GEMM path requires H (M) and msl (N) at least as large as the largest tile
// variant (32x128); smaller shapes use the naive fallback kernel instead.
constexpr int MIN_M_GEMM = 32;
constexpr int MIN_N_GEMM = 128;

constexpr int kMinWorkgroupsPerSubslice = 4;

// Picks (global_range, local_range) for a 2D kernel (the paged-K
// gather and naive fallback kernels below, both indexed as (batch, kv
// position)).
std::pair<sycl::range<2>, sycl::range<2>> compute_2d_launch_ranges(int dim0, int dim1) {
  int64_t sub_group_size = std::max<int64_t>(1, dpcppMaxSubGroupSize());
  int64_t target_workgroups = dpcppGpuSubsliceCount() * kMinWorkgroupsPerSubslice;

  int local1 = std::min(dim1, WG_SIZE);
  while (local1 > sub_group_size) {
    int64_t workgroups = static_cast<int64_t>(dim0) * div_up(dim1, local1);
    if (workgroups >= target_workgroups) break;
    local1 = std::max<int>(static_cast<int>(sub_group_size), local1 / 2);
  }

  sycl::range<2> local_range(1, local1);
  int global_1 = div_up(dim1, local1) * local1;
  sycl::range<2> global_range(dim0, global_1);
  return {global_range, local_range};
}

// Total (batch, position) pairs below which the Reduce kernel's per-thread
// H-head serial load chain leaves too few work-items to fill the device's
// subslices. Below this, split the head loop across multiple cooperating work-items
// instead of running with a single lane per output element. Tuned empirically.
constexpr int64_t kMinReduceElemsForNoSplit = 4096;

// Pick how many work-items cooperate on each output element's head-reduction:
// 1 (no split) once B*msl alone gives the device enough independent work,
// otherwise the largest divisor of H in {8,4,2} to shorten the per-lane
// serial load chain without leaving remainder heads unhandled.
inline int select_reduce_heads_per_group(int64_t total_elems, int H) {
  if (total_elems > kMinReduceElemsForNoSplit) return 1;
  for (int cand : {8, 4, 2}) {
    if (H % cand == 0) return cand;
  }
  return 1;
}

// Launches Fp8PagedMqaLogitsReduceKernel. Each work-group covers
// `positions_per_group` (batch,position) outputs, each reduced by
// `heads_per_group` cooperating work-items, so the *total* work-group size
// stays ~WG_SIZE regardless of heads_per_group.
void launch_reduce_kernel(
    sycl::queue& queue,
    const float* dots_ptr,
    const float* weights_ptr,
    const float* k_scale_ptr,
    const int32_t* seq_lens_ptr,
    float* out_ptr,
    int B,
    int H,
    int max_seq_len,
    int heads_per_group) {
  int positions_per_group = std::max(1, WG_SIZE / heads_per_group);
  positions_per_group = std::min(positions_per_group, max_seq_len);
  sycl::range<3> local_range(1, positions_per_group, heads_per_group);
  int global_1 = div_up(max_seq_len, positions_per_group) * positions_per_group;
  sycl::range<3> global_range(B, global_1, heads_per_group);
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> partials_slm(sycl::range<1>(positions_per_group * heads_per_group), cgh);
    nsa::Fp8PagedMqaLogitsReduceKernel kernel{
        dots_ptr, weights_ptr, k_scale_ptr, seq_lens_ptr, out_ptr, B, H, max_seq_len, heads_per_group, partials_slm};
    cgh.parallel_for(sycl::nd_range<3>(global_range, local_range), kernel);
  });
}

// GEMM tile-shape variants (M, N, K), ordered from largest (best per-tile
// arithmetic intensity/reuse) to smallest (most workgroups). All are valid
// CTA tile shapes for MMA_Atom<XE_8x16x16_F32F16F16F32_TT> with subgroup
// layout (1,4,1): M must be a multiple of 8, N a multiple of 64, K a
// multiple of 16 (see cute::TiledMMAHelper's CanonicalBlockShape).
using GemmTileShapeLarge = cute::Shape<cute::_32, cute::_128, cute::_32>;
using GemmTileShapeMedium = cute::Shape<cute::_16, cute::_64, cute::_32>;
using GemmTileShapeSmall = cute::Shape<cute::_8, cute::_64, cute::_32>;

// Subgroup (WarpLayout) partitioning of each CTA tile
using GemmSGLayoutWide =
    cute::Layout<cute::Shape<cute::_1, cute::_4, cute::_1>, cute::Stride<cute::_4, cute::_1, cute::_0>>;
using GemmSGLayoutNarrow =
    cute::Layout<cute::Shape<cute::_1, cute::_2, cute::_1>, cute::Stride<cute::_2, cute::_1, cute::_0>>;

// Tuned for the GEMM part.
constexpr int64_t kMinOccupancyTiles = 64;

template <typename GemmTileShape>
inline bool is_tile_aligned(int M, int N, int K) {
  using namespace cute;
  return M % get<0>(GemmTileShape{}) == 0 && N % get<1>(GemmTileShape{}) == 0 && K % get<2>(GemmTileShape{}) == 0;
}

template <typename GemmTileShape>
inline int64_t total_tiles(int batch, int M, int N) {
  using namespace cute;
  return static_cast<int64_t>(batch) * (M / get<0>(GemmTileShape{})) * (N / get<1>(GemmTileShape{}));
}

// Batched FP8 GEMM via SYCL-TLA: for each b, D_b(M,N) = A_b(M,K) @ B_b(N,K)^T.
// Falls back to torch::bmm if not tile-aligned.
void fp8_gemm_xe20_batched_inplace(
    sycl::queue& queue,
    const torch::Tensor& a_fp8,  // (batch, M, K) fp8
    const torch::Tensor& b_fp8,  // (batch, N, K)
    torch::Tensor& d_f32,        // (batch, M, N) output
    int batch,
    int M,
    int N,
    int K,
    int64_t a_batch_stride,
    int64_t b_batch_stride,
    int64_t d_batch_stride,
    at::Device device) {
  auto a_ptr = a_fp8.data_ptr<uint8_t>();
  auto b_ptr = b_fp8.data_ptr<uint8_t>();
  auto d_ptr = d_f32.data_ptr<float>();

  // Pick the largest tile that still clears the minimum-occupancy target;
  // fall back to progressively smaller (but still aligned) tiles if the
  // large one would under-fill the device, and to the smallest aligned tile
  // if none reach the target (maximize occupancy as a last resort).
  if (is_tile_aligned<GemmTileShapeLarge>(M, N, K) &&
      total_tiles<GemmTileShapeLarge>(batch, M, N) >= kMinOccupancyTiles) {
    nsa::fp8_mqa_gemm_batched_launch<GemmTileShapeLarge, GemmSGLayoutWide>(
        &queue, a_ptr, b_ptr, d_ptr, batch, M, N, K, a_batch_stride, b_batch_stride, d_batch_stride);
    return;
  }
  if (is_tile_aligned<GemmTileShapeMedium>(M, N, K) &&
      total_tiles<GemmTileShapeMedium>(batch, M, N) >= kMinOccupancyTiles) {
    nsa::fp8_mqa_gemm_batched_launch<GemmTileShapeMedium, GemmSGLayoutNarrow>(
        &queue, a_ptr, b_ptr, d_ptr, batch, M, N, K, a_batch_stride, b_batch_stride, d_batch_stride);
    return;
  }
  if (is_tile_aligned<GemmTileShapeSmall>(M, N, K)) {
    nsa::fp8_mqa_gemm_batched_launch<GemmTileShapeSmall, GemmSGLayoutNarrow>(
        &queue, a_ptr, b_ptr, d_ptr, batch, M, N, K, a_batch_stride, b_batch_stride, d_batch_stride);
    return;
  }
  if (is_tile_aligned<GemmTileShapeMedium>(M, N, K)) {
    nsa::fp8_mqa_gemm_batched_launch<GemmTileShapeMedium, GemmSGLayoutNarrow>(
        &queue, a_ptr, b_ptr, d_ptr, batch, M, N, K, a_batch_stride, b_batch_stride, d_batch_stride);
    return;
  }
  if (is_tile_aligned<GemmTileShapeLarge>(M, N, K)) {
    nsa::fp8_mqa_gemm_batched_launch<GemmTileShapeLarge, GemmSGLayoutWide>(
        &queue, a_ptr, b_ptr, d_ptr, batch, M, N, K, a_batch_stride, b_batch_stride, d_batch_stride);
    return;
  }

  // Fallback to torch::bmm if dimensions are not tile-aligned by any variant.
  // a_fp8 and b_fp8 may have arbitrary leading dims from the call site;
  // reshape to 3D (batch, M/N, K) for bmm compatibility.
  auto a_bf16 = a_fp8.reshape({batch, M, K}).to(at::ScalarType::BFloat16);
  auto b_bf16 = b_fp8.reshape({batch, N, K}).to(at::ScalarType::BFloat16);
  auto result = at::bmm(a_bf16, b_bf16.transpose(1, 2)).to(at::ScalarType::Float);
  d_f32.copy_(result.reshape(d_f32.sizes()));
}

}  // namespace

// fp8_paged_mqa_logits: decode path
SGL_KERNEL_EXPORT torch::Tensor fp8_paged_mqa_logits(
    const torch::Tensor& q_fp8,
    const torch::Tensor& kv_cache,
    const torch::Tensor& weights,
    const torch::Tensor& seq_lens,
    const torch::Tensor& block_tables,
    const std::optional<torch::Tensor>& schedule_metadata,
    int64_t max_seq_len,
    bool clean_logits) {
  TORCH_CHECK(!schedule_metadata.has_value(), "fp8_paged_mqa_logits does not support schedule_metadata on XPU");
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

  int B = q_fp8.size(0);
  int H = q_fp8.size(2);
  int D = q_fp8.size(3);
  int page_size = kv_cache.size(1);
  int head_dim_with_sf = kv_cache.size(3);
  int max_num_blocks = block_tables.size(1);
  TORCH_CHECK(
      max_seq_len >= 0 && max_seq_len <= std::numeric_limits<int>::max(),
      "max_seq_len exceeds int32 range for XPU kernel");
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
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == B && weights.size(1) == H, "weights must be (B, H)");

  // clean_logits is accepted for API compatibility but output is always
  // zero-initialized — the cost is negligible relative to GEMM.
  (void)clean_logits;
  auto logits = torch::zeros({B, msl}, torch::dtype(torch::kFloat32).device(q_fp8.device()));
  if (B == 0 || msl == 0) return logits;

  auto seq_lens_flat = seq_lens.dim() == 2 ? seq_lens.contiguous().view({-1}) : seq_lens.contiguous();
  TORCH_CHECK(
      seq_lens_flat.size(0) == B,
      "seq_lens must have B elements after flattening, got ",
      seq_lens_flat.size(0),
      " vs B=",
      B);

  // Ensure contiguity
  auto q_contig = q_fp8.contiguous();
  auto kv_contig = kv_cache.contiguous();
  auto weights_contig = weights.contiguous();
  auto block_tables_contig = block_tables.contiguous();

  auto stream = c10::xpu::getCurrentXPUStream();
  auto& queue = stream.queue();

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  GPU_Clock timer;
  timer.start();
#endif

  bool use_gemm = (H >= MIN_M_GEMM && msl >= MIN_N_GEMM);

  if (use_gemm) {
    // Stage 1/2 temporaries (k_gathered: B*msl*D bytes, dots: B*H*msl*4 bytes)
    // scale with the *full* decode batch B.
    // Chunk over the batch dimension so peak temporary memory is bounded by a
    // fixed budget, mirroring the batch-slicing done for the FMHA prefill score workspace.
    constexpr int64_t kDefaultChunkBudgetBytes = 512LL * 1024 * 1024;  // 512 MiB
    static const int64_t chunk_budget_bytes = [] {
      if (const char* env = std::getenv("SGL_KERNEL_FP8_PAGED_MQA_CHUNK_MB")) {
        int64_t mb = std::atoll(env);
        if (mb > 0) return mb << 20;
      }
      return kDefaultChunkBudgetBytes;
    }();
    // Per-batch bytes for the two chunked temporaries: k_gathered (msl*D uint8) + dots (H*msl*4 float32) +
    // k_scale_gathered (msl*4 float32).
    int64_t per_batch_bytes = static_cast<int64_t>(H) * msl * sizeof(float) + static_cast<int64_t>(msl) * D +
                              static_cast<int64_t>(msl) * sizeof(float);
    int chunk_b = static_cast<int>(std::max<int64_t>(1, chunk_budget_bytes / std::max<int64_t>(per_batch_bytes, 1)));
    chunk_b = std::min(chunk_b, B);
    if (std::getenv("SGL_KERNEL_FP8_PAGED_MQA_VERBOSE") != nullptr) {
      TORCH_WARN_ONCE(
          "fp8_paged_mqa_logits: B=",
          B,
          " H=",
          H,
          " msl=",
          msl,
          " -> chunk_b=",
          chunk_b,
          " (",
          div_up(B, chunk_b),
          " chunks)");
    }

    // To avoid OOM, slice the batch dimension into chunks of size chunk_b and process each chunk independently.
    // Hard to fuse the gather + GEMM + reduction into a single kernel because the kernels have different launch
    // configs.
    for (int start = 0; start < B; start += chunk_b) {
      int cur_b = std::min(chunk_b, B - start);

      auto q_chunk = q_contig.narrow(0, start, cur_b);
      auto seq_lens_chunk = seq_lens_flat.narrow(0, start, cur_b);
      auto block_tables_chunk = block_tables_contig.narrow(0, start, cur_b);
      auto weights_chunk = weights_contig.narrow(0, start, cur_b);
      auto logits_chunk = logits.narrow(0, start, cur_b);

      // Stage 1: Gather K from pages into contiguous buffer (sized for this chunk only)
      // Out of bound positions are filled with zeros in the kernel.
      auto k_gathered =
          torch::empty({static_cast<int64_t>(cur_b) * msl, D}, torch::dtype(torch::kUInt8).device(q_fp8.device()));
      auto k_scale_gathered = torch::empty({cur_b, msl}, torch::dtype(torch::kFloat32).device(q_fp8.device()));

      nsa::PagedKGatherKernel gather_kernel{
          kv_contig.data_ptr<uint8_t>(),
          block_tables_chunk.data_ptr<int32_t>(),
          seq_lens_chunk.data_ptr<int32_t>(),
          k_gathered.data_ptr<uint8_t>(),
          k_scale_gathered.data_ptr<float>(),
          cur_b,
          D,
          page_size,
          max_num_blocks,
          msl,
          head_dim_with_sf};
      auto [gather_global_range, gather_local_range] = compute_2d_launch_ranges(cur_b, msl);
      sycl_kernel_submit(gather_global_range, gather_local_range, queue, gather_kernel);

      // Stage 2: Batched SYCL-TLA FP8 GEMM — all cur_b GEMMs in this chunk in one launch.
      // TODO: switch to torch._scaled_grouped_mm once it is implemented on XPU.
      auto dots = torch::empty({cur_b, H, msl}, torch::dtype(torch::kFloat32).device(q_fp8.device()));

      fp8_gemm_xe20_batched_inplace(
          queue,
          q_chunk,     // (cur_b,1,H,D), A batch stride = H*D
          k_gathered,  // (cur_b*msl,D), B batch stride = msl*D
          dots,        // (cur_b,H,msl), D batch stride = H*msl
          cur_b,
          H,
          msl,
          D,
          static_cast<int64_t>(H) * D,
          static_cast<int64_t>(msl) * D,
          static_cast<int64_t>(H) * msl,
          q_fp8.device());

      // Stage 3: Reduction. Split the head loop across multiple cooperating
      // work-items per output element when B*msl alone is too small to fill
      // the device (see select_reduce_heads_per_group).
      int heads_per_group = select_reduce_heads_per_group(static_cast<int64_t>(cur_b) * msl, H);
      launch_reduce_kernel(
          queue,
          dots.data_ptr<float>(),
          weights_chunk.data_ptr<float>(),
          k_scale_gathered.data_ptr<float>(),
          seq_lens_chunk.data_ptr<int32_t>(),
          logits_chunk.data_ptr<float>(),
          cur_b,
          H,
          msl,
          heads_per_group);
    }

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
    // Q @ K^T over paged fp8 KV: 2 * B * H * msl * D flops for the matmul + reduce (~B*H*msl).
    const double flops =
        2.0 * static_cast<double>(B_next) * static_cast<double>(H) * static_cast<double>(msl) * static_cast<double>(D);
    const double bytes = static_cast<double>(q_fp8.numel()) * 1.0 +
                         static_cast<double>(B_next) * static_cast<double>(msl) * (static_cast<double>(D) + 4.0) +
                         static_cast<double>(B_next) * static_cast<double>(H) * 4.0 +
                         static_cast<double>(logits.numel()) * 4.0;
    ::sglkernel::report_kernel_perf("fp8_paged_mqa_logits", queue, timer, bytes, flops);
#endif

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
      B,
      H,
      D,
      page_size,
      max_num_blocks,
      msl};
  auto [paged_global_range, paged_local_range] = compute_2d_launch_ranges(B, msl);
  sycl_kernel_submit(paged_global_range, paged_local_range, queue, paged_kernel);

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  // Q @ K^T over paged fp8 KV: 2 * B * H * msl * D flops for the matmul + reduce (~B*H*msl).
  const double flops =
      2.0 * static_cast<double>(B_next) * static_cast<double>(H) * static_cast<double>(msl) * static_cast<double>(D);
  const double bytes = static_cast<double>(q_fp8.numel()) * 1.0 +
                       static_cast<double>(B_next) * static_cast<double>(msl) * (static_cast<double>(D) + 4.0) +
                       static_cast<double>(B_next) * static_cast<double>(H) * 4.0 +
                       static_cast<double>(logits.numel()) * 4.0;
  ::sglkernel::report_kernel_perf("fp8_paged_mqa_logits", queue, timer, bytes, flops);
#endif

  return logits;
}

#undef SYCL_INTEL_TARGET
