#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>

#include "Utils.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "kernels/moe/xe20/moe_kernel.hpp"

using namespace cute;

using ElementAccumulator = float;  // <- data type of accumulator

template <typename Tile, typename SGLayout, int ActType, bool FuseAct, bool WithBias>
void Xe20MoEGEMMLauncher(
    sycl::queue q,
    const void* activations,
    const void* weights,
    const void* scales,
    const void* bias,
    void* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* num_rows_per_expert_device,
    const int num_experts,
    int* workspace);

using Tile_8_64_32 = Shape<_8, _64, _32>;
using Tile_16_64_32 = Shape<_16, _64, _32>;
using Tile_32_64_32 = Shape<_32, _64, _32>;
using Tile_128_64_32 = Shape<_128, _64, _32>;
using Tile_128_128_32 = Shape<_128, _128, _32>;
using Tile_256_64_32 = Shape<_256, _64, _32>;
using Tile_256_256_32 = Shape<_256, _256, _32>;

using SG_1_4_1 = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
using SG_4_2_1 = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;
using SG_8_2_1 = Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>;
using SG_8_4_1 = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

#define DECLARE_XE20_MOE_EXTERN(Tile, SGLayout, ActType, FuseAct, WithBias)             \
  extern template void Xe20MoEGEMMLauncher<Tile, SGLayout, ActType, FuseAct, WithBias>( \
      sycl::queue,                                                                      \
      const void*,                                                                      \
      const void*,                                                                      \
      const void*,                                                                      \
      const void*,                                                                      \
      void*,                                                                            \
      const int,                                                                        \
      const int,                                                                        \
      const int*,                                                                       \
      const int,                                                                        \
      int*);

#define DECLARE_XE20_MOE_TILE_ALL_FUSES(Tile, SGLayout)    \
  DECLARE_XE20_MOE_EXTERN(Tile, SGLayout, 0, true, true)   \
  DECLARE_XE20_MOE_EXTERN(Tile, SGLayout, 0, true, false)  \
  DECLARE_XE20_MOE_EXTERN(Tile, SGLayout, 0, false, true)  \
  DECLARE_XE20_MOE_EXTERN(Tile, SGLayout, 0, false, false) \
  DECLARE_XE20_MOE_EXTERN(Tile, SGLayout, 1, true, true)   \
  DECLARE_XE20_MOE_EXTERN(Tile, SGLayout, 1, true, false)  \
  DECLARE_XE20_MOE_EXTERN(Tile, SGLayout, 1, false, true)  \
  DECLARE_XE20_MOE_EXTERN(Tile, SGLayout, 1, false, false)

#define DECLARE_XE20_MOE_TILE_FUSE(Tile, SGLayout, FuseAct)  \
  DECLARE_XE20_MOE_EXTERN(Tile, SGLayout, 0, FuseAct, true)  \
  DECLARE_XE20_MOE_EXTERN(Tile, SGLayout, 0, FuseAct, false) \
  DECLARE_XE20_MOE_EXTERN(Tile, SGLayout, 1, FuseAct, true)  \
  DECLARE_XE20_MOE_EXTERN(Tile, SGLayout, 1, FuseAct, false)

DECLARE_XE20_MOE_TILE_ALL_FUSES(Tile_8_64_32, SG_1_4_1)
DECLARE_XE20_MOE_TILE_ALL_FUSES(Tile_16_64_32, SG_1_4_1)
DECLARE_XE20_MOE_TILE_ALL_FUSES(Tile_32_64_32, SG_1_4_1)
DECLARE_XE20_MOE_TILE_FUSE(Tile_128_64_32, SG_4_2_1, true)
DECLARE_XE20_MOE_TILE_FUSE(Tile_128_128_32, SG_4_2_1, false)
DECLARE_XE20_MOE_TILE_FUSE(Tile_256_64_32, SG_8_2_1, true)
DECLARE_XE20_MOE_TILE_FUSE(Tile_256_256_32, SG_8_4_1, false)

#undef DECLARE_XE20_MOE_TILE_FUSE
#undef DECLARE_XE20_MOE_TILE_ALL_FUSES
#undef DECLARE_XE20_MOE_EXTERN

#define LAUNCH_MOE(...)                       \
  Xe20MoEGEMMLauncher<__VA_ARGS__>(           \
      queue,                                  \
      activations.data_ptr(),                 \
      weights.data_ptr(),                     \
      nullptr,                                \
      bias_ptr,                               \
      output.data_ptr(),                      \
      gemm_n,                                 \
      gemm_k,                                 \
      total_rows_for_experts.data_ptr<int>(), \
      n_experts,                              \
      atomic_buffer.data_ptr<int>())

#define DISPATCH_MOE_HELPER_BIAS(ActType, FuseAct, WithBias, ...) \
  do {                                                            \
    if (WithBias) {                                               \
      LAUNCH_MOE(__VA_ARGS__, ActType, FuseAct, true);            \
    } else {                                                      \
      LAUNCH_MOE(__VA_ARGS__, ActType, FuseAct, false);           \
    }                                                             \
  } while (0)

#define DISPATCH_MOE_HELPER_FUSE_ACT(ActType, FuseAct, WithBias, ...)  \
  do {                                                                 \
    if (FuseAct) {                                                     \
      DISPATCH_MOE_HELPER_BIAS(ActType, true, WithBias, __VA_ARGS__);  \
    } else {                                                           \
      DISPATCH_MOE_HELPER_BIAS(ActType, false, WithBias, __VA_ARGS__); \
    }                                                                  \
  } while (0)

#define DISPATCH_MOE_HELPER_ACT_TYPE(ActType, FuseAct, WithBias, ...)    \
  do {                                                                   \
    switch (ActType) {                                                   \
      case 0:                                                            \
        DISPATCH_MOE_HELPER_FUSE_ACT(0, FuseAct, WithBias, __VA_ARGS__); \
        break;                                                           \
      case 1:                                                            \
        DISPATCH_MOE_HELPER_FUSE_ACT(1, FuseAct, WithBias, __VA_ARGS__); \
        break;                                                           \
      default:                                                           \
        TORCH_CHECK(false, "Unsupported activation type");               \
    }                                                                    \
  } while (0)

#define DISPATCH_MOE(ActType, FuseAct, WithBias, ...) \
  DISPATCH_MOE_HELPER_ACT_TYPE(ActType, FuseAct, WithBias, __VA_ARGS__)

void moe_grouped_mm_nt_xe20(
    torch::Tensor& output,
    const torch::Tensor& activations,
    const torch::Tensor& weights,
    const std::optional<at::Tensor>& bias,
    const torch::Tensor& total_rows_for_experts,
    const int64_t n_experts,
    const int64_t activation_type,  // 0=silu, 1=gelu
    bool fuse_act) {
  int total_m = activations.sizes()[0];
  int gemm_k = activations.sizes()[1];
  auto weights_shape = weights.sizes().vec();
  int gemm_n = weights.sizes()[1];
  int avg_m = total_m / n_experts;

  TORCH_CHECK(weights_shape.size() == 3, "weights must be 3D");
  TORCH_CHECK(weights_shape[0] == n_experts, "weights must have n_experts as the first dimension");
  TORCH_CHECK(weights_shape[1] == gemm_n, "weights must be gemm_n * gemm_k");
  TORCH_CHECK(
      weights_shape[0] == total_rows_for_experts.size(0),
      "rows_for_experts must have the same size as the first dimension of weights");
  TORCH_CHECK(output.sizes()[0] == total_m, "output must have the same number of rows as activations");
  if (fuse_act) {
    TORCH_CHECK(output.sizes()[1] == gemm_n / 2, "output must have half the number of columns as activations");
  } else {
    TORCH_CHECK(output.sizes()[1] == gemm_n, "output must have the same number of columns as activations");
  }
  TORCH_CHECK(n_experts % 8 == 0, "n_experts must be a multiple of 8 for the current implementation");
  TORCH_CHECK(
      activations.scalar_type() == weights.scalar_type(), "activations and weights must have the same data type");
  TORCH_CHECK(
      activations.scalar_type() == at::ScalarType::BFloat16,
      "Only bfloat16 are supported in moe_grouped_mm_nt currently");
  if (bias.has_value()) {
    TORCH_CHECK(bias->dim() == 2, "bias must be 2D [n_experts, N]");
    TORCH_CHECK(bias->size(0) == n_experts && bias->size(1) == gemm_n, "bias shape mismatch with weight");
  }

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  at::Tensor atomic_buffer = at::empty({static_cast<long>(1)}, activations.options().dtype(at::kInt));
  bool with_bias = bias.has_value();
  void* bias_ptr = with_bias ? bias->data_ptr() : nullptr;
  bool small_weight = (int64_t)gemm_k * gemm_n <= (int64_t)4096 * 4096;  // heuristic for small K*N, can be tuned

  if (avg_m <= 8) {
    DISPATCH_MOE(
        activation_type, fuse_act, with_bias, Shape<_8, _64, _32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>);
  } else if (avg_m <= 16 && small_weight) {
    DISPATCH_MOE(
        activation_type, fuse_act, with_bias, Shape<_16, _64, _32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>);
  } else if (avg_m <= 32 && small_weight) {
    DISPATCH_MOE(
        activation_type, fuse_act, with_bias, Shape<_32, _64, _32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>);
  } else if (avg_m <= 128 && small_weight) {
    if (fuse_act) {
      DISPATCH_MOE(
          activation_type, true, with_bias, Shape<_128, _64, _32>, Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>);
    } else {
      DISPATCH_MOE(
          activation_type, false, with_bias, Shape<_128, _128, _32>, Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>);
    }
  } else {
    if (fuse_act) {
      DISPATCH_MOE(
          activation_type, true, with_bias, Shape<_256, _64, _32>, Layout<Shape<_8, _2, _1>, Stride<_2, _1, _0>>);
    } else {
      DISPATCH_MOE(
          activation_type, false, with_bias, Shape<_256, _256, _32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>);
    }
  }
}

#undef SYCL_INTEL_TARGET
