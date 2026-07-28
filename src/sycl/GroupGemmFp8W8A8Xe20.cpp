#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>

#include "Utils.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "kernels/moe/xe20/fp8_w8a8/moe_kernel.hpp"

using namespace cute;

template <typename Tile, typename SGLayout, int ActType, bool FuseAct, bool WithBias>
void Xe20MoEGEMMFp8W8A8Launcher(
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
    int ld_b);

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
//   - Tile_128_64_32 (fuse_act) / Tile_128_128_32 (no fuse_act) as the
//     single "large" tier for now, instead of the bf16 menu's 128-tier
//     *and* 256-tier split by avg_m/small_weight. This is a placeholder:
//     whether the 128-tile is actually the best choice for large-avg_m fp8
//     traffic (vs. re-adding a 256-tile once its register pressure is
//     known, or vs. a differently-shaped tile e.g. taller-K) is an open
//     tuning question, not a considered decision.
using Tile_8_64_32 = Shape<_8, _64, _32>;
using Tile_16_64_32 = Shape<_16, _64, _32>;
using Tile_32_64_32 = Shape<_32, _64, _32>;
using Tile_128_64_32 = Shape<_128, _64, _32>;
using Tile_128_128_32 = Shape<_128, _128, _32>;

using SG_1_4_1 = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
using SG_4_2_1 = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;

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
      int);

// v1 only instantiates ActType=0 (SILU). This is NOT a "fp8 can only do
// SILU" design limitation - moe_mainloop.hpp's fused-activation epilogue
// already generically supports GELU/SWIGLU_GPT_OSS/SWIGLU_DEEPSEEK_V4 via
// the same moe_xe20::apply_fused_activation<ActType> used by bf16/MXFP4.
// It is a deliberately narrow *instantiation* choice: unlike MXFP4 (tied
// to specific checkpoint families), fp8 is a general-purpose quant format
// used across many model families (grok/gemma4 -> GELU, gpt_oss ->
// SWIGLU_GPT_OSS, deepseek_v2/minimax_m3/step3p5 -> SWIGLU_DEEPSEEK_V4),
// so which of those need fp8 support is a product decision, not something
// to infer from the MXFP4 precedent. Add the other ActType values here
// (and to GroupGemmFp8W8A8Xe20.cmake) once the target fp8 checkpoints are
// known - see xpu_fp8_moe_minimal_plan.md.
#define DECLARE_XE20_MOE_FP8_TILES_B(WithBias)                             \
  DECLARE_XE20_MOE_FP8_EXTERN(Tile_8_64_32, SG_1_4_1, 0, true, WithBias)   \
  DECLARE_XE20_MOE_FP8_EXTERN(Tile_8_64_32, SG_1_4_1, 0, false, WithBias)  \
  DECLARE_XE20_MOE_FP8_EXTERN(Tile_16_64_32, SG_1_4_1, 0, true, WithBias)  \
  DECLARE_XE20_MOE_FP8_EXTERN(Tile_16_64_32, SG_1_4_1, 0, false, WithBias) \
  DECLARE_XE20_MOE_FP8_EXTERN(Tile_32_64_32, SG_1_4_1, 0, true, WithBias)  \
  DECLARE_XE20_MOE_FP8_EXTERN(Tile_32_64_32, SG_1_4_1, 0, false, WithBias) \
  DECLARE_XE20_MOE_FP8_EXTERN(Tile_128_64_32, SG_4_2_1, 0, true, WithBias) \
  DECLARE_XE20_MOE_FP8_EXTERN(Tile_128_128_32, SG_4_2_1, 0, false, WithBias)

DECLARE_XE20_MOE_FP8_TILES_B(false)
DECLARE_XE20_MOE_FP8_TILES_B(true)

#undef DECLARE_XE20_MOE_FP8_TILES_B
#undef DECLARE_XE20_MOE_FP8_EXTERN

#define LAUNCH_MOE_FP8(...)                   \
  Xe20MoEGEMMFp8W8A8Launcher<__VA_ARGS__>(    \
      queue,                                  \
      activations.data_ptr(),                 \
      act_scales.data_ptr(),                  \
      weights.data_ptr(),                     \
      weight_scales.data_ptr(),               \
      bias_ptr,                               \
      output.data_ptr(),                      \
      gemm_n,                                 \
      gemm_k,                                 \
      total_rows_for_experts.data_ptr<int>(), \
      n_experts,                              \
      atomic_buffer.data_ptr<int>(),          \
      static_cast<float>(gemm1_alpha),        \
      static_cast<float>(gemm1_limit),        \
      ld_b)

#define LAUNCH_MOE_FP8_BIAS(FuseAct, WithBias, ...)   \
  do {                                                \
    if (WithBias) {                                   \
      LAUNCH_MOE_FP8(__VA_ARGS__, 0, FuseAct, true);  \
    } else {                                          \
      LAUNCH_MOE_FP8(__VA_ARGS__, 0, FuseAct, false); \
    }                                                 \
  } while (0)

#define DISPATCH_MOE_FP8(ActType, FuseAct, WithBias, ...)                                   \
  do {                                                                                      \
    TORCH_CHECK(                                                                            \
        (ActType) == 0,                                                                     \
        "fp8 w8a8 fused MoE kernel built with ActType=0 (silu) only for now; got ActType=", \
        (ActType),                                                                          \
        ". See src/GroupGemmFp8W8A8Xe20.cmake to add more ActType instantiations.");        \
    if (FuseAct) {                                                                          \
      LAUNCH_MOE_FP8_BIAS(true, WithBias, __VA_ARGS__);                                     \
    } else {                                                                                \
      LAUNCH_MOE_FP8_BIAS(false, WithBias, __VA_ARGS__);                                    \
    }                                                                                       \
  } while (0)

// FP8 (E4M3) W8A8 MoE grouped GEMM. `activations`/`weights` are
// float8_e4m3fn raw bytes. `act_scales` is a per-token (per-M-row) fp32
// direct multiplier. `weight_scales` is a per-(N-row, K-group) fp32 direct
// multiplier with FP8_GROUP_SIZE_K (128) elements per K-group - see
// moe_mainloop.hpp. A genuinely 2-D-blocked (e.g. DeepSeek 128x128)
// weight-scale tensor must be pre-expanded to per-N-row by the caller
// (python/sgl_kernel/moe.py does this via repeat_interleave) before
// reaching this op; per-tensor (single-scalar) weight scale is not
// supported by this first version.
void moe_grouped_mm_nt_xe20_fp8_w8a8(
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
  int total_m = activations.sizes()[0];
  int gemm_k = activations.sizes()[1];
  auto weights_shape = weights.sizes().vec();
  int gemm_n = weights.sizes()[1];
  int avg_m = total_m / n_experts;

  TORCH_CHECK(weights_shape.size() == 3, "weights must be 3D [E, N, K]");
  TORCH_CHECK(weights_shape[0] == n_experts, "weights must have n_experts as the first dimension");
  TORCH_CHECK(weights_shape[1] == gemm_n, "weights must be gemm_n * gemm_k");
  TORCH_CHECK(weights_shape[2] == gemm_k, "weights last dim must equal K (fp8 is 1 byte/element, no packing)");
  TORCH_CHECK(
      weights_shape[0] == total_rows_for_experts.size(0),
      "rows_for_experts must have the same size as the first dimension of weights");
  TORCH_CHECK(output.sizes()[0] == total_m, "output must have the same number of rows as activations");
  if (fuse_act) {
    TORCH_CHECK(output.sizes()[1] == gemm_n / 2, "output must have half the number of columns when fuse_act is true");
  } else {
    TORCH_CHECK(output.sizes()[1] == gemm_n, "output must have the same number of columns as activations");
  }
  TORCH_CHECK(n_experts % 8 == 0, "n_experts must be a multiple of 8 for the current implementation");

  TORCH_CHECK(activations.scalar_type() == at::ScalarType::Float8_e4m3fn, "activations must be float8_e4m3fn");
  TORCH_CHECK(weights.scalar_type() == at::ScalarType::Float8_e4m3fn, "weights must be float8_e4m3fn");
  TORCH_CHECK(act_scales.scalar_type() == at::kFloat, "act_scales must be float32");
  TORCH_CHECK(act_scales.dim() == 1 && act_scales.size(0) == total_m, "act_scales must be 1D [M_total]");
  TORCH_CHECK(weight_scales.scalar_type() == at::kFloat, "weight_scales must be float32 (direct multiplier)");
  TORCH_CHECK(weight_scales.dim() == 3, "weight_scales must be 3D [E, N, K/128]");
  TORCH_CHECK(weight_scales.size(0) == n_experts, "weight_scales expert dim mismatch");
  TORCH_CHECK(
      weight_scales.size(1) == gemm_n, "weight_scales must be pre-expanded to one row per N (see comment above)");
  TORCH_CHECK(
      gemm_k % 128 == 0 && weight_scales.size(2) == gemm_k / 128,
      "weight_scales last dim must equal K/128 (FP8_GROUP_SIZE_K=128)");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16, "output must be bfloat16");

  if (bias.has_value()) {
    TORCH_CHECK(bias->scalar_type() == at::kFloat, "moe_grouped_mm_nt_xe20_fp8_w8a8: bias must be float32");
    TORCH_CHECK(bias->dim() == 2, "bias must be 2D [n_experts, N]");
    TORCH_CHECK(bias->size(0) == n_experts && bias->size(1) == gemm_n, "bias shape mismatch with weight");
  }

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  at::Tensor atomic_buffer = at::empty({static_cast<long>(1)}, activations.options().dtype(at::kInt));
  bool with_bias = bias.has_value();
  void* bias_ptr = with_bias ? bias->data_ptr() : nullptr;
  int ld_b = static_cast<int>(weights.stride(1));

  // No "unfused GEMM1 for huge-weight/small-M" heuristic yet (see
  // moe_kernel.hpp header comment) - tile choice is purely a function of
  // avg_m for this first version.
  if (avg_m <= 8) {
    DISPATCH_MOE_FP8(activation_type, fuse_act, with_bias, Tile_8_64_32, SG_1_4_1);
  } else if (avg_m <= 16) {
    DISPATCH_MOE_FP8(activation_type, fuse_act, with_bias, Tile_16_64_32, SG_1_4_1);
  } else if (avg_m <= 32) {
    DISPATCH_MOE_FP8(activation_type, fuse_act, with_bias, Tile_32_64_32, SG_1_4_1);
  } else {
    if (fuse_act) {
      DISPATCH_MOE_FP8(activation_type, true, with_bias, Tile_128_64_32, SG_4_2_1);
    } else {
      DISPATCH_MOE_FP8(activation_type, false, with_bias, Tile_128_128_32, SG_4_2_1);
    }
  }
}

#undef SYCL_INTEL_TARGET
