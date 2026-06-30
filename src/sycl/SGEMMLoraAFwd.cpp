#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>
#include <type_traits>
#include <vector>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "kernels/lora/group_gemm_lora_launcher.hpp"

//----------------- Kernel launch (host-side argument preparation) --------------------//
//
// Builds the per-segment problem_sizes, pointer arrays, and stride arrays that
// the CUTLASS pointer-array grouped GEMM consumes, then dispatches to the
// templated launcher in sgemm_grouped_launcher.hpp.
//
// For each segment s in [0, num_segments):
//   M_s = seg_indptr[s+1] - seg_indptr[s]
//   N   = stack_num * max_rank                (uniform across segments)
//   K   = input_x.size(1) = input_dim         (uniform across segments)
//
//   a_ptr[s] = input_x.data_ptr()  + seg_indptr[s] * K              (byte-offset)
//   b_ptr[s] = weights.data_ptr()  + weight_indices[s] * N * K      (byte-offset)
//   d_ptr[s] = output.data_ptr()   + seg_indptr[s] * N              (byte-offset)
//
// Per option (a) we trust that weight rows beyond lora_ranks[lora] are zero,
// so we compute the full M_s x N output for every segment; no rank masking.
//
template <typename TensorDType>
void launch_sgemm_lora_a_fwd(
    const torch::Tensor& input_x,
    const torch::Tensor& weights,
    const torch::Tensor& seg_indptr_i32,
    const torch::Tensor& weight_indices_i32,
    torch::Tensor& output,
    const int stack_num,
    const int max_rank,
    const int num_segments,
    sycl::queue& queue) {
  const int K = static_cast<int>(input_x.size(1));   // input_dim
  const int N = stack_num * max_rank;                 // output columns
  const int64_t elem_bytes = static_cast<int64_t>(sizeof(TensorDType));

  auto seg_indptr_cpu = seg_indptr_i32.cpu().contiguous();
  auto weight_indices_cpu = weight_indices_i32.cpu().contiguous();
  const int32_t* seg_indptr_h = seg_indptr_cpu.data_ptr<int32_t>();
  const int32_t* weight_indices_h = weight_indices_cpu.data_ptr<int32_t>();

  // ---- Build per-segment arrays on the host ----
  std::vector<int32_t> problem_sizes_h(static_cast<size_t>(num_segments) * 3);
  std::vector<int64_t> a_ptrs_h(num_segments);
  std::vector<int64_t> b_ptrs_h(num_segments);
  std::vector<int64_t> d_ptrs_h(num_segments);

  const int64_t a_base = reinterpret_cast<int64_t>(input_x.data_ptr());
  const int64_t b_base = reinterpret_cast<int64_t>(weights.data_ptr());
  const int64_t d_base = reinterpret_cast<int64_t>(output.data_ptr());
  const int64_t b_per_lora_bytes = static_cast<int64_t>(N) * K * elem_bytes;

  for (int s = 0; s < num_segments; ++s) {
    const int32_t row_start = seg_indptr_h[s];
    const int32_t M_s = seg_indptr_h[s + 1] - row_start;
    const int32_t lora_id = weight_indices_h[s];

    problem_sizes_h[3 * s + 0] = M_s;
    problem_sizes_h[3 * s + 1] = N;
    problem_sizes_h[3 * s + 2] = K;

    a_ptrs_h[s] = a_base + static_cast<int64_t>(row_start) * K * elem_bytes;
    b_ptrs_h[s] = b_base + static_cast<int64_t>(lora_id) * b_per_lora_bytes;
    d_ptrs_h[s] = d_base + static_cast<int64_t>(row_start) * N * elem_bytes;
  }

  // Strides in elements 
  std::vector<int64_t> stride_A_h(num_segments, static_cast<int64_t>(K));
  std::vector<int64_t> stride_B_h(num_segments, static_cast<int64_t>(K));
  std::vector<int64_t> stride_D_h(num_segments, static_cast<int64_t>(N));

  // Move host-built arrays to the XPU device.
  auto device = input_x.device();
  auto cpu_i32 = torch::TensorOptions().dtype(torch::kInt32);
  auto cpu_i64 = torch::TensorOptions().dtype(torch::kInt64);

  auto problem_sizes =
      torch::from_blob(problem_sizes_h.data(), {num_segments, 3}, cpu_i32).clone().to(device);
  auto a_ptrs  = torch::from_blob(a_ptrs_h.data(),  {num_segments}, cpu_i64).clone().to(device);
  auto b_ptrs  = torch::from_blob(b_ptrs_h.data(),  {num_segments}, cpu_i64).clone().to(device);
  auto d_ptrs  = torch::from_blob(d_ptrs_h.data(),  {num_segments}, cpu_i64).clone().to(device);
  auto stride_A = torch::from_blob(stride_A_h.data(), {num_segments}, cpu_i64).clone().to(device);
  auto stride_B = torch::from_blob(stride_B_h.data(), {num_segments}, cpu_i64).clone().to(device);
  auto stride_D = torch::from_blob(stride_D_h.data(), {num_segments}, cpu_i64).clone().to(device);

  //----------------- Compile-time perf knobs (LoRA-A forward) -----------------//
  // All perf-critical types are pinned here at the call site rather than inside
  // the shared launcher, so each LoRA entry point (A-fwd, B-fwd, QKV-B-fwd, ...)
  // can pick its own tile / subgroup / MMA-atom / copy-atom mix without forking
  // the launcher.
  //
  // Knobs that are currently uniform across input dtypes (tunable here):
  //   TileShape    = 256 x 256 x 32   -- canonical bf16 tile from upstream
  //                                      04_bmg_grouped_gemm; K=32 is also valid
  //                                      for fp32-via-TF32 (MMA K=8 divides 32).
  //   ThreadLayout = 8 x 4 x 1        -- 32 subgroups per workgroup.
  //   LayoutB      = ColumnMajor      -- free-transpose B; the B atom does the real transpose.
  //   PipelineStages = 2              -- matches the upstream BMG reference.
  //
  // Per-dtype knobs (selected below via `std::conditional_t`):
  //   MmaAtom: which XMX MMA instruction the mainloop calls. Pulled from
  //            `XeMmaAtomFor<>` for the canonical pick; can be swapped for a
  //            smaller-M variant in the same family (XE_{4,2,1}x16x{8,16}_*)
  //            if the TileShape M is small or sub-grouping needs to change.
  //   Copy atoms: see comment block below.
  //
  // Per-dtype copy atoms (selected via `std::conditional_t` below):
  //   bf16/fp16 (16-bit storage):
  //     A: XE_2D_U16x32x32_LD_N           (row-major load)
  //     B: XE_2D_U16x16x16_LD_T           (transpose-at-load, K=16)
  //     C: XE_2D_U32x8x16_LD_N            (fp32 acc; inert when beta=0)
  //     D: XE_2D_U16x8x16_ST_N            (16-bit narrow store)
  //   fp32-via-TF32 (32-bit storage; final result kept as full fp32):
  //     A: XE_2D_TF32x32x16_LD_N          (TF32 sub-blocks matching MMA A-side ALayout)
  //     B: XE_2D_U32x16x8_LD_T            (transpose-at-load, K=8)
  //     C: XE_2D_U32x8x16_LD_N            (same as bf16/fp16 -- C is fp32 either way)
  //     D: XE_2D_U32x8x16_ST_N            (32-bit store -- no narrowing back to TF32)
  //
  // Alternative pairings to try here later (not currently used):
  //   bf16/fp16 with LayoutB = RowMajor + B atom = XE_2D_U16x32x32_LD_V (VNNI),
  //   smaller-M MMA variants (XE_{4,2,1}x16x{8,16}_*), different TileShape per dtype, etc.
  using TileShape    = cute::Shape<cute::_256, cute::_256, cute::_32>;
  using ThreadLayout = cute::Layout<
      cute::Shape <cute::_8, cute::_4, cute::_1>,
      cute::Stride<cute::_4, cute::_1, cute::_0>>;
  using LayoutB      = cutlass::layout::ColumnMajor;
  constexpr int PipelineStages = 2;

  // Per-dtype MMA atom. Canonical pick via XeMmaAtomFor<>
  using ElementCutlass = typename at::native::xpu::ToCutlassElementType<TensorDType>::type;
  using MmaAtom        = typename at::native::xpu::XeMmaAtomFor<ElementCutlass>::type;

  // Per-dtype gmem copy-atom selection
  using GmemTiledCopyA = std::conditional_t<
      std::is_same_v<TensorDType, float>,
      cute::XE_2D_U32x32x16_LD_N,
      cute::XE_2D_U16x32x32_LD_N>;
  using GmemTiledCopyB = std::conditional_t<
      std::is_same_v<TensorDType, float>,
      cute::XE_2D_U32x16x8_LD_T,
      cute::XE_2D_U16x16x16_LD_T>;
  using GmemTiledCopyC = cute::XE_2D_U32x8x16_LD_N;
  using GmemTiledCopyD = std::conditional_t<
      std::is_same_v<TensorDType, float>,
      cute::XE_2D_U32x8x16_ST_N,
      cute::XE_2D_U16x8x16_ST_N>;

  // Dispatch to the CUTLASS pointer-array grouped GEMM launcher.
  at::native::xpu::launch_group_gemm_lora_fwd<
      TensorDType,
      MmaAtom,
      TileShape,
      ThreadLayout,
      GmemTiledCopyA,
      GmemTiledCopyB,
      GmemTiledCopyC,
      GmemTiledCopyD,
      LayoutB,
      PipelineStages>(
      queue,
      problem_sizes,
      a_ptrs,
      b_ptrs,
      /*c_ptrs   =*/d_ptrs,
      d_ptrs,
      stride_A,
      stride_B,
      /*stride_C =*/stride_D,
      stride_D,
      num_segments,
      /*alpha    =*/1.0f,
      /*beta     =*/0.0f);
}


//----------------- Main API function --------------------//

void sgemm_lora_a_fwd(
    torch::Tensor& output,           // [num_tokens, max_rank]
    const torch::Tensor& input_x,    // [num_tokens, input_dim]
    const torch::Tensor& weights,    // [num_loras, stack_num*max_rank, input_dim]
    const int64_t stack_num,
    const torch::Tensor& seg_indptr,                       // [num_segments + 1,]
    const torch::Tensor& weight_indices,                   // [num_segments,]
    const torch::Tensor& lora_ranks,                       // [num_loras,]
    const std::optional<torch::Tensor>&
        seg_lens  // [num_segments,] optional; currently unused, reserved for future per-segment optimizations
) {
  CHECK_INPUT(input_x);
  CHECK_INPUT(weights);
  CHECK_INPUT(seg_indptr);
  CHECK_INPUT(weight_indices);
  CHECK_INPUT(lora_ranks);
  CHECK_INPUT(output);

  TORCH_CHECK(input_x.dim() == 2, "input_x must be a 2D tensor");
  TORCH_CHECK(weights.dim() == 3, "weights must be a 3D tensor");
  TORCH_CHECK(seg_indptr.dim() == 1, "seg_indptr must be a 1D tensor");
  TORCH_CHECK(weight_indices.dim() == 1, "weight_indices must be a 1D tensor");
  TORCH_CHECK(lora_ranks.dim() == 1, "lora_ranks must be a 1D tensor");
  TORCH_CHECK(output.dim() == 2, "output must be a 2D tensor");

  const int64_t num_loras_i64 = weights.size(0);
  const int64_t max_rank_i64 = weights.size(1) / stack_num;
  const int64_t num_tokens_i64 = input_x.size(0);

  TORCH_CHECK(lora_ranks.numel() == num_loras_i64, "lora_ranks.numel() must equal weights.size(0)");
  TORCH_CHECK(num_loras_i64 > 0, "weights.size(0) and lora_ranks.numel() must be greater than 0");
  TORCH_CHECK(
      num_tokens_i64 == 0 || seg_indptr.numel() >= 2, "seg_indptr must have at least 2 elements when num_tokens > 0");
  const int64_t num_segments_i64 = seg_indptr.numel() - 1;
  TORCH_CHECK(weight_indices.numel() == num_segments_i64, "weight_indices.numel() must equal seg_indptr.numel() - 1");
  if (num_segments_i64 > 0) {
    auto [min_wi, max_wi] = torch::aminmax(weight_indices);
    TORCH_CHECK(
        min_wi.item<int64_t>() >= 0 && max_wi.item<int64_t>() < num_loras_i64,
        "weight_indices values must be in [0, weights.size(0))");
  }
  // Validate output tensor size and dtype
  TORCH_CHECK(
      output.size(0) == num_tokens_i64 && output.size(1) == max_rank_i64 * stack_num,
      "Output tensor must have shape (num_tokens, max_rank * stack_num)");
  TORCH_CHECK(output.scalar_type() == weights.scalar_type(), "Output tensor dtype must match weights dtype");
  TORCH_CHECK(weights.scalar_type() == input_x.scalar_type(), "Input tensor dtype must match weights dtype");
  if (num_tokens_i64 == 0) {
    return;
  }

  TORCH_CHECK(seg_indptr[0].item<int64_t>() == 0, "seg_indptr[0] must be 0");
  TORCH_CHECK(
      seg_indptr[seg_indptr.numel() - 1].item<int64_t>() == num_tokens_i64, "seg_indptr[-1] must equal num_tokens");
  auto seg_len_tensor = seg_indptr.slice(0, 1) - seg_indptr.slice(0, 0, seg_indptr.size(0) - 1);
  auto [seg_len_min, seg_len_max] = torch::aminmax(seg_len_tensor);
  TORCH_CHECK(seg_len_min.item<int>() >= 0, "seg_indptr must be non-decreasing");
  (void)seg_len_max;  // not needed: grouped GEMM handles variable M per group

  auto [min_lr, max_lr] = torch::aminmax(lora_ranks);
  TORCH_CHECK(
      min_lr.item<int64_t>() >= 0 && max_lr.item<int>() <= max_rank_i64,
      "All values in lora_ranks must be within the range [0, max_rank]");

  // Cast index tensors to int32 for the host-side metadata read in the launcher.
  auto seg_indptr_i32 = seg_indptr.scalar_type() == torch::kInt32 ? seg_indptr : seg_indptr.to(torch::kInt32);
  auto weight_indices_i32 =
      weight_indices.scalar_type() == torch::kInt32 ? weight_indices : weight_indices.to(torch::kInt32);

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  const int max_rank = static_cast<int>(max_rank_i64);
  const int num_segments = static_cast<int>(num_segments_i64);
  const int stack_num_ = static_cast<int>(stack_num);

  // Dispatch kernel based on data type
  if (weights.scalar_type() == torch::kFloat32) {
    TORCH_CHECK(false, "Float32 is not supported. Use bfloat16 or float16.");
  } else if (weights.scalar_type() == torch::kHalf) {
    launch_sgemm_lora_a_fwd<at::Half>(
        input_x, weights, seg_indptr_i32, weight_indices_i32, output,
        stack_num_, max_rank, num_segments, queue);
  } else if (weights.scalar_type() == torch::kBFloat16) {
    launch_sgemm_lora_a_fwd<at::BFloat16>(
        input_x, weights, seg_indptr_i32, weight_indices_i32, output,
        stack_num_, max_rank, num_segments, queue);
  } else {
    TORCH_CHECK(false, "Unsupported data type for weights");
  }
}
