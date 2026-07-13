#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>

#include "Utils.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "kernels/moe/xe20/mxfp4_w4a16/moe_kernel.hpp"

using namespace cute;

template <typename Tile, typename SGLayout, int ActType, bool FuseAct, bool WithBias>
void Xe20MoEGEMMMxfp4W4A16W4A16Launcher(
    sycl::queue q,
    const void* activations,
    const void* packed_weights,
    const void* scales,
    const void* bias,
    void* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* num_rows_per_expert_device,
    const int num_experts,
    int* workspace,
    float gemm1_alpha,
    float gemm1_limit);

// Tile menu (see GroupGemmMxfp4W4A16Xe20.cmake for the full instantiation matrix):
//   - avg_m ≤ 8:                  <_8, _64, _32>, SG_1_4_1
//   - avg_m ≤ 128, fuse_act:      <_128, _64, _32>, SG_4_2_1
//   - avg_m ≤ 128, no fuse_act:   <_128, _128, _32>, SG_4_2_1
//   - avg_m >  128, fuse_act:     <_256, _64, _32>, SG_8_2_1
//   - avg_m >  128, no fuse_act:  <_256, _256, _32>, SG_8_4_1
using Tile_8_64_32 = Shape<_8, _64, _32>;
using Tile_128_64_32 = Shape<_128, _64, _32>;
using Tile_128_128_32 = Shape<_128, _128, _32>;
using Tile_256_64_32 = Shape<_256, _64, _32>;
using Tile_256_256_32 = Shape<_256, _256, _32>;

using SG_1_4_1 = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
using SG_4_2_1 = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;
using SG_8_2_1 = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
using SG_8_4_1 = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

#define DECLARE_XE20_MOE_MXFP4_EXTERN(Tile, SGLayout, ActType, FuseAct, WithBias)                      \
  extern template void Xe20MoEGEMMMxfp4W4A16W4A16Launcher<Tile, SGLayout, ActType, FuseAct, WithBias>( \
      sycl::queue,                                                                                     \
      const void*,                                                                                     \
      const void*,                                                                                     \
      const void*,                                                                                     \
      const void*,                                                                                     \
      void*,                                                                                           \
      const int,                                                                                       \
      const int,                                                                                       \
      const int*,                                                                                      \
      const int,                                                                                       \
      int*,                                                                                            \
      float,                                                                                           \
      float);

// Declare the instantiations GroupGemmMxfp4W4A16Xe20.cmake emits:
// ActType ∈ {0 silu, 2 swiglu_gpt_oss, 4 swiglu_deepseek_v4} × WithBias ∈
// {false, true} × the fuse_act tile menu. The dispatcher below allows the same
// set at runtime. Keep the two sides in sync (see the cmake header comment for
// the TP>1 module-pressure caveat).
#define DECLARE_XE20_MOE_MXFP4_TILES_B(ActType, WithBias)                            \
  DECLARE_XE20_MOE_MXFP4_EXTERN(Tile_8_64_32, SG_1_4_1, ActType, true, WithBias)     \
  DECLARE_XE20_MOE_MXFP4_EXTERN(Tile_8_64_32, SG_1_4_1, ActType, false, WithBias)    \
  DECLARE_XE20_MOE_MXFP4_EXTERN(Tile_128_64_32, SG_4_2_1, ActType, true, WithBias)   \
  DECLARE_XE20_MOE_MXFP4_EXTERN(Tile_128_128_32, SG_4_2_1, ActType, false, WithBias) \
  DECLARE_XE20_MOE_MXFP4_EXTERN(Tile_256_64_32, SG_8_2_1, ActType, true, WithBias)   \
  DECLARE_XE20_MOE_MXFP4_EXTERN(Tile_256_256_32, SG_8_4_1, ActType, false, WithBias)

#define DECLARE_XE20_MOE_MXFP4_TILES(ActType)    \
  DECLARE_XE20_MOE_MXFP4_TILES_B(ActType, false) \
  DECLARE_XE20_MOE_MXFP4_TILES_B(ActType, true)

DECLARE_XE20_MOE_MXFP4_TILES(0)
DECLARE_XE20_MOE_MXFP4_TILES(2)
DECLARE_XE20_MOE_MXFP4_TILES(4)

#undef DECLARE_XE20_MOE_MXFP4_TILES
#undef DECLARE_XE20_MOE_MXFP4_TILES_B
#undef DECLARE_XE20_MOE_MXFP4_EXTERN

#define LAUNCH_MOE_MXFP4(...)                      \
  Xe20MoEGEMMMxfp4W4A16W4A16Launcher<__VA_ARGS__>( \
      queue,                                       \
      activations.data_ptr(),                      \
      packed_weights.data_ptr(),                   \
      scales.data_ptr(),                           \
      bias_ptr,                                    \
      output.data_ptr(),                           \
      gemm_n,                                      \
      gemm_k,                                      \
      total_rows_for_experts.data_ptr<int>(),      \
      n_experts,                                   \
      atomic_buffer.data_ptr<int>(),               \
      static_cast<float>(gemm1_alpha),             \
      static_cast<float>(gemm1_limit))

// Dispatch over the instantiated matrix: ActType ∈ {0 silu, 2 swiglu_gpt_oss,
// 4 swiglu_deepseek_v4} × WithBias ∈ {false, true} × FuseAct. Selects the
// concrete template specialization for the runtime (ActType, FuseAct,
// WithBias) triple. Combos outside this set hit the TORCH_CHECK below; restore
// them in both this file and the cmake matrix before rebuilding.
#define LAUNCH_MOE_MXFP4_ACT(ActTypeLit, FuseAct, WithBias, ...) \
  do {                                                           \
    if (FuseAct) {                                               \
      if (WithBias) {                                            \
        LAUNCH_MOE_MXFP4(__VA_ARGS__, ActTypeLit, true, true);   \
      } else {                                                   \
        LAUNCH_MOE_MXFP4(__VA_ARGS__, ActTypeLit, true, false);  \
      }                                                          \
    } else {                                                     \
      if (WithBias) {                                            \
        LAUNCH_MOE_MXFP4(__VA_ARGS__, ActTypeLit, false, true);  \
      } else {                                                   \
        LAUNCH_MOE_MXFP4(__VA_ARGS__, ActTypeLit, false, false); \
      }                                                          \
    }                                                            \
  } while (0)

#define DISPATCH_MOE_MXFP4(ActType, FuseAct, WithBias, ...)                                                    \
  do {                                                                                                         \
    TORCH_CHECK(                                                                                               \
        (ActType) == 0 || (ActType) == 2 || (ActType) == 4,                                                    \
        "mxfp4 fused kernel built with ActType=0 (silu), 2 (swiglu_gpt_oss) and 4 (swiglu_deepseek_v4) only; " \
        "got ActType=",                                                                                        \
        (ActType),                                                                                             \
        ". See src/GroupGemmMxfp4W4A16Xe20.cmake to re-enable additional ActType/WithBias combos.");           \
    if ((ActType) == 4) {                                                                                      \
      LAUNCH_MOE_MXFP4_ACT(4, FuseAct, WithBias, __VA_ARGS__);                                                 \
    } else if ((ActType) == 2) {                                                                               \
      LAUNCH_MOE_MXFP4_ACT(2, FuseAct, WithBias, __VA_ARGS__);                                                 \
    } else {                                                                                                   \
      LAUNCH_MOE_MXFP4_ACT(0, FuseAct, WithBias, __VA_ARGS__);                                                 \
    }                                                                                                          \
  } while (0)

void moe_grouped_mm_nt_xe20_mxfp4_w4a16(
    torch::Tensor& output,
    const torch::Tensor& activations,
    const torch::Tensor& packed_weights,  // [E, N, K/2] int8
    const torch::Tensor& scales,          // [E, N, K/32] float32 direct multiplier (N-outer)
    const std::optional<at::Tensor>& bias,
    const torch::Tensor& total_rows_for_experts,
    const int64_t n_experts,
    const int64_t activation_type,
    bool fuse_act,
    double gemm1_alpha,
    double gemm1_limit) {
  int total_m = activations.sizes()[0];
  int gemm_k = activations.sizes()[1];
  auto pw_shape = packed_weights.sizes().vec();
  int gemm_n = pw_shape[1];

  TORCH_CHECK(pw_shape.size() == 3, "packed_weights must be 3D [E, N, K/2]");
  TORCH_CHECK(pw_shape[0] == n_experts, "packed_weights first dim must equal n_experts");
  TORCH_CHECK(pw_shape[1] == gemm_n, "packed_weights second dim must equal N");
  TORCH_CHECK(pw_shape[2] == gemm_k / 2, "packed_weights last dim must equal K/2 (two E2M1 per byte)");
  TORCH_CHECK(packed_weights.scalar_type() == at::ScalarType::Char, "packed_weights must be int8");

  auto sc_shape = scales.sizes().vec();
  TORCH_CHECK(sc_shape.size() == 3, "scales must be 3D [E, N, K/32]");
  TORCH_CHECK(sc_shape[0] == n_experts, "scales first dim must equal n_experts");
  TORCH_CHECK(sc_shape[1] == gemm_n, "scales second dim must equal N");
  TORCH_CHECK(sc_shape[2] == gemm_k / MoE_MXFP4_W4A16::MXFP4_GROUP_SIZE, "scales last dim must equal K/32");
  TORCH_CHECK(scales.scalar_type() == at::ScalarType::Float, "scales must be float32 (direct multiplier)");

  TORCH_CHECK(
      n_experts == total_rows_for_experts.size(0),
      "total_rows_for_experts must have the same size as packed_weights.size(0)");
  TORCH_CHECK(output.sizes()[0] == total_m, "output rows must match activations rows");
  if (fuse_act) {
    TORCH_CHECK(output.sizes()[1] == gemm_n / 2, "output must have N/2 columns when fuse_act is true");
  } else {
    TORCH_CHECK(output.sizes()[1] == gemm_n, "output must have N columns when fuse_act is false");
  }

  TORCH_CHECK(n_experts % 8 == 0, "n_experts must be a multiple of 8 for the current implementation");
  TORCH_CHECK(activations.scalar_type() == at::ScalarType::BFloat16, "activations must be bfloat16");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16, "output must be bfloat16");
  TORCH_CHECK(gemm_k % MoE_MXFP4_W4A16::MXFP4_GROUP_SIZE == 0, "K must be a multiple of GROUP_SIZE=32");

  if (bias.has_value()) {
    TORCH_CHECK(bias->scalar_type() == at::kFloat, "bias must be float32");
    TORCH_CHECK(bias->dim() == 2, "bias must be 2D [E, N]");
    TORCH_CHECK(bias->size(0) == n_experts && bias->size(1) == gemm_n, "bias shape mismatch");
  }

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  at::Tensor atomic_buffer = at::empty({static_cast<long>(1)}, activations.options().dtype(at::kInt));
  bool with_bias = bias.has_value();
  void* bias_ptr = with_bias ? bias->data_ptr() : nullptr;

  int avg_m = total_m / n_experts;
  bool small_weight = (int64_t)gemm_k * gemm_n <= MOE_GROUPED_GEMM_SMALL_WEIGHT_THRESHOLD;

  if (avg_m <= 8) {
    DISPATCH_MOE_MXFP4(
        activation_type, fuse_act, with_bias, Shape<_8, _64, _32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>);
  } else if (avg_m <= 128 && small_weight) {
    if (fuse_act) {
      DISPATCH_MOE_MXFP4(
          activation_type, true, with_bias, Shape<_128, _64, _32>, Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>);
    } else {
      DISPATCH_MOE_MXFP4(
          activation_type, false, with_bias, Shape<_128, _128, _32>, Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>);
    }
  } else {
    if (fuse_act) {
      DISPATCH_MOE_MXFP4(
          activation_type, true, with_bias, Shape<_256, _64, _32>, Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>);
    } else {
      DISPATCH_MOE_MXFP4(
          activation_type, false, with_bias, Shape<_256, _256, _32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>);
    }
  }
}

#undef SYCL_INTEL_TARGET
