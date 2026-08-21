#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>

#include "Utils.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "kernels/moe/xe20/fp8_w8a8/moe_kernel.hpp"
#include "sgl_kernel_export.h"

using namespace cute;

template <typename Tile, typename SGLayout, int ActType, bool FuseAct, bool WithBias>
__attribute__((visibility("default"))) void Xe20MoEGEMMFp8W8A8Launcher(
    sycl::queue q,
    const void* activations,
    const void* act_scales,
    const void* weights,
    const void* weight_scales,
    const void* bias,
    void* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* num_rows_per_expert_device,
    const int num_experts,
    int* workspace,
    float gemm1_alpha,
    float gemm1_limit,
    int ld_b,
    bool act_scale_grouped,
    int act_scale_k_groups);

// Tile menu (deliberately smaller/more conservative than the bf16 and
// MXFP4 W4A16 menus - see moe_mainloop.hpp header comment). This mainloop
// decodes BOTH A and B from fp8 to fp16 in registers (bf16/MXFP4 only ever
// decode one operand, if any), so register pressure per work-item is
// higher for a given tile size. Concretely, compared to the bf16 menu:
//   - No 256-wide tiles (Tile_256_64_32 / Tile_256_256_32) yet. These are
//     the bf16 menu's largest-register-footprint tiles; omitted here
//     until occupancy/spill behavior for the double-decode path has been
//     profiled. Add them back (mirroring GroupGemmXe20.cmake) once tuned.
//   - Same small-M tiles as bf16/MXFP4 (_8/_16/_32 x _64, SG_1_4_1) since
//     those already have the smallest register footprint of the existing
//     menus and are unlikely to regress.
//   - Tile_32_64_32 remains the largest tile for now. A Xe2 A/B measurement
//     showed it faster than the initial Tile128 variants for wider-N and
//     larger-M shapes; the initial Tile128 variants are not instantiated in
//     this matrix until a narrow-N workload justifies their binary cost.
using Tile_8_64_32 = Shape<_8, _64, _32>;
using Tile_16_64_32 = Shape<_16, _64, _32>;
using Tile_32_64_32 = Shape<_32, _64, _32>;

using SG_1_4_1 = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;

#define DECLARE_XE20_MOE_FP8_EXTERN(Tile, SGLayout, ActType, FuseAct, WithBias)                \
  extern template void Xe20MoEGEMMFp8W8A8Launcher<Tile, SGLayout, ActType, FuseAct, WithBias>( \
      sycl::queue,                                                                             \
      const void*,                                                                             \
      const void*,                                                                             \
      const void*,                                                                             \
      const void*,                                                                             \
      const void*,                                                                             \
      void*,                                                                                   \
      const int,                                                                               \
      const int,                                                                               \
      const int*,                                                                              \
      const int,                                                                               \
      int*,                                                                                    \
      float,                                                                                   \
      float,                                                                                   \
      int,                                                                                     \
      bool,                                                                                    \
      int);

#define DECLARE_XE20_MOE_FP8_BIAS_VARIANTS(Tile, SGLayout, ActType, FuseAct) \
  DECLARE_XE20_MOE_FP8_EXTERN(Tile, SGLayout, ActType, FuseAct, false)       \
  DECLARE_XE20_MOE_FP8_EXTERN(Tile, SGLayout, ActType, FuseAct, true)

#define DECLARE_XE20_MOE_FP8_TILES                                      \
  DECLARE_XE20_MOE_FP8_BIAS_VARIANTS(Tile_8_64_32, SG_1_4_1, 0, false)  \
  DECLARE_XE20_MOE_FP8_BIAS_VARIANTS(Tile_16_64_32, SG_1_4_1, 0, false) \
  DECLARE_XE20_MOE_FP8_BIAS_VARIANTS(Tile_32_64_32, SG_1_4_1, 0, false)

DECLARE_XE20_MOE_FP8_TILES

#undef DECLARE_XE20_MOE_FP8_TILES
#undef DECLARE_XE20_MOE_FP8_BIAS_VARIANTS
#undef DECLARE_XE20_MOE_FP8_EXTERN

#define LAUNCH_MOE_FP8(WithBias, ...)                \
  Xe20MoEGEMMFp8W8A8Launcher<__VA_ARGS__, WithBias>( \
      queue,                                         \
      activations.data_ptr(),                        \
      act_scales.data_ptr(),                         \
      weights.data_ptr(),                            \
      weight_scales.data_ptr(),                      \
      bias_ptr,                                      \
      output.data_ptr(),                             \
      gemm_n,                                        \
      gemm_k,                                        \
      total_rows_for_experts.data_ptr<int>(),        \
      n_experts,                                     \
      atomic_buffer.data_ptr<int>(),                 \
      static_cast<float>(gemm1_alpha),               \
      static_cast<float>(gemm1_limit),               \
      ld_b,                                          \
      act_scales.dim() == 2,                         \
      act_scales.dim() == 2 ? static_cast<int>(act_scales.size(1)) : 1)

// Activation is always external to the FP8 GEMM. Keep the public arguments for
// API compatibility, but dispatch only the activation-neutral ActType=0 path.
#define DISPATCH_MOE_FP8(...)                       \
  do {                                              \
    if (bias.has_value()) {                         \
      LAUNCH_MOE_FP8(true, __VA_ARGS__, 0, false);  \
    } else {                                        \
      LAUNCH_MOE_FP8(false, __VA_ARGS__, 0, false); \
    }                                               \
  } while (0)

// FP8 (E4M3) W8A8 MoE grouped GEMM. `activations`/`weights` are
// float8_e4m3fn raw bytes. `act_scales` is either legacy [M] per-token or
// production [M, K/128] per-token-group fp32 direct multipliers.
// `weight_scales` is a per-(N-row, K-group) fp32 direct multiplier with
// FP8_GROUP_SIZE_K (128) elements per K-group - see
// moe_mainloop.hpp. A genuinely 2-D-blocked (e.g. DeepSeek 128x128)
// weight-scale tensor must be pre-expanded to per-N-row by the caller
// (python/sgl_kernel/moe.py does this via repeat_interleave) before
// reaching this op; per-tensor (single-scalar) weight scale is not
// supported by this first version.
SGL_KERNEL_EXPORT void moe_grouped_mm_nt_xe20_fp8_w8a8(
    torch::Tensor& output,
    const torch::Tensor& activations,
    const torch::Tensor& act_scales,
    const torch::Tensor& weights,
    const torch::Tensor& weight_scales,
    const std::optional<at::Tensor>& bias,
    const torch::Tensor& total_rows_for_experts,
    const int64_t n_experts,
    const int64_t activation_type,
    bool fuse_act,
    double gemm1_alpha,
    double gemm1_limit) {
  CHECK_INPUT(output);
  CHECK_INPUT(activations);
  CHECK_INPUT(act_scales);
  CHECK_INPUT(weights);
  CHECK_INPUT(weight_scales);
  CHECK_INPUT(total_rows_for_experts);
  TORCH_CHECK(n_experts > 0, "n_experts must be positive");
  TORCH_CHECK(output.device() == activations.device(), "output must be on the same device as activations");
  TORCH_CHECK(act_scales.device() == activations.device(), "act_scales must be on the same device as activations");
  TORCH_CHECK(weights.device() == activations.device(), "weights must be on the same device as activations");
  TORCH_CHECK(
      weight_scales.device() == activations.device(), "weight_scales must be on the same device as activations");
  TORCH_CHECK(
      total_rows_for_experts.device() == activations.device(),
      "total_rows_for_experts must be on the same device as activations");
  if (bias.has_value()) {
    const auto& bias_tensor = *bias;
    CHECK_INPUT(bias_tensor);
    TORCH_CHECK(bias_tensor.device() == activations.device(), "bias must be on the same device as activations");
  }
  TORCH_CHECK(activations.dim() == 2, "activations must be 2D [M_total, K]");
  TORCH_CHECK(output.dim() == 2, "output must be 2D [M_total, N]");
  TORCH_CHECK(total_rows_for_experts.dim() == 1, "total_rows_for_experts must be 1D [E]");
  int total_m = activations.sizes()[0];
  int gemm_k = activations.sizes()[1];
  auto weights_shape = weights.sizes().vec();
  TORCH_CHECK(weights_shape.size() == 3, "weights must be 3D [E, N, K]");
  int gemm_n = weights.sizes()[1];
  int avg_m = total_m / n_experts;
  int tile_n = 64;

  TORCH_CHECK(!fuse_act, "FP8 W8A8 MoE requires activation outside GEMM1");

  TORCH_CHECK(weights_shape[0] == n_experts, "weights must have n_experts as the first dimension");
  TORCH_CHECK(weights_shape[2] == gemm_k, "weights last dim must equal K (fp8 is 1 byte/element, no packing)");
  TORCH_CHECK(
      weights_shape[0] == total_rows_for_experts.size(0),
      "rows_for_experts must have the same size as the first dimension of weights");
  TORCH_CHECK(output.sizes()[0] == total_m, "output must have the same number of rows as activations");
  TORCH_CHECK(output.sizes()[1] == gemm_n, "output must have the same number of columns as activations");
  TORCH_CHECK(
      gemm_n % tile_n == 0,
      "FP8 W8A8 MoE requires the output width to be divisible by the selected tile width (output width=",
      gemm_n,
      ", tile width=",
      tile_n,
      "); non-aligned N is not supported yet");
  TORCH_CHECK(n_experts % 8 == 0, "n_experts must be a multiple of 8 for the current implementation");

  TORCH_CHECK(activations.scalar_type() == at::ScalarType::Float8_e4m3fn, "activations must be float8_e4m3fn");
  TORCH_CHECK(weights.scalar_type() == at::ScalarType::Float8_e4m3fn, "weights must be float8_e4m3fn");
  TORCH_CHECK(act_scales.scalar_type() == at::kFloat, "act_scales must be float32");
  TORCH_CHECK(
      (act_scales.dim() == 1 || act_scales.dim() == 2) && act_scales.size(0) == total_m,
      "act_scales must be [M_total] or [M_total, K/128]");
  if (act_scales.dim() == 2) {
    TORCH_CHECK(
        gemm_k % 128 == 0 && act_scales.size(1) == gemm_k / 128, "grouped act_scales must have shape [M_total, K/128]");
  }
  TORCH_CHECK(weight_scales.scalar_type() == at::kFloat, "weight_scales must be float32 (direct multiplier)");
  TORCH_CHECK(weight_scales.dim() == 3, "weight_scales must be 3D [E, N, K/128]");
  TORCH_CHECK(weight_scales.size(0) == n_experts, "weight_scales expert dim mismatch");
  TORCH_CHECK(
      weight_scales.size(1) == gemm_n, "weight_scales must be pre-expanded to one row per N (see comment above)");
  TORCH_CHECK(
      gemm_k % 128 == 0 && weight_scales.size(2) == gemm_k / 128,
      "weight_scales last dim must equal K/128 (FP8_GROUP_SIZE_K=128)");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16, "output must be bfloat16");
  TORCH_CHECK(total_rows_for_experts.scalar_type() == at::ScalarType::Int, "total_rows_for_experts must be int32");
  if (bias.has_value()) {
    TORCH_CHECK(bias->scalar_type() == at::kFloat, "bias must be float32");
    TORCH_CHECK(bias->dim() == 2, "bias must be 2D [E, N]");
    TORCH_CHECK(bias->size(0) == n_experts && bias->size(1) == gemm_n, "bias shape must be [E, N]");
  }

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  at::Tensor atomic_buffer = at::empty({static_cast<long>(1)}, activations.options().dtype(at::kInt));
  void* bias_ptr = bias.has_value() ? bias->data_ptr() : nullptr;
  int ld_b = static_cast<int>(weights.stride(1));

  if (avg_m <= 8) {
    DISPATCH_MOE_FP8(Tile_8_64_32, SG_1_4_1);
  } else if (avg_m <= 16) {
    DISPATCH_MOE_FP8(Tile_16_64_32, SG_1_4_1);
  } else {
    DISPATCH_MOE_FP8(Tile_32_64_32, SG_1_4_1);
  }
}

#undef SYCL_INTEL_TARGET
