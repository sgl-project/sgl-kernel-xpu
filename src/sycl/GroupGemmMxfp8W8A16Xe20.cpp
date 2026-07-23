#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>

#include "Utils.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "kernels/moe/xe20/mxfp8_w8a16/moe_kernel.hpp"

using namespace cute;

template <typename Tile, typename SGLayout, int ActType, bool FuseAct, bool WithBias>
void Xe20MoEGEMMMxfp8W8A16Launcher(
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

// Tile menu (see GroupGemmMxfp8W8A16Xe20.cmake for the instantiation matrix).
// The dense mxfp8 W8A16 linear path is non-fused (ActType=0, FuseAct=false),
// so only that slice is instantiated -- with and without bias. Tile selection
// mirrors the mxfp4 non-fused tiles keyed on avg_m.
using Tile_8_64_32 = Shape<_8, _64, _32>;
using Tile_128_128_32 = Shape<_128, _128, _32>;
using Tile_256_256_32 = Shape<_256, _256, _32>;

using SG_1_4_1 = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
using SG_4_2_1 = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;
using SG_8_4_1 = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;

#define DECLARE_XE20_MOE_MXFP8_EXTERN(Tile, SGLayout, WithBias)                             \
  extern template void Xe20MoEGEMMMxfp8W8A16Launcher<Tile, SGLayout, 0, false, WithBias>(   \
      sycl::queue,                                                                          \
      const void*,                                                                          \
      const void*,                                                                          \
      const void*,                                                                          \
      const void*,                                                                          \
      void*,                                                                                \
      const int,                                                                            \
      const int,                                                                            \
      const int*,                                                                           \
      const int,                                                                            \
      int*,                                                                                 \
      float,                                                                                \
      float);

#define DECLARE_XE20_MOE_MXFP8_TILES(WithBias)              \
  DECLARE_XE20_MOE_MXFP8_EXTERN(Tile_8_64_32, SG_1_4_1, WithBias)     \
  DECLARE_XE20_MOE_MXFP8_EXTERN(Tile_128_128_32, SG_4_2_1, WithBias)  \
  DECLARE_XE20_MOE_MXFP8_EXTERN(Tile_256_256_32, SG_8_4_1, WithBias)

DECLARE_XE20_MOE_MXFP8_TILES(false)
DECLARE_XE20_MOE_MXFP8_TILES(true)

#undef DECLARE_XE20_MOE_MXFP8_TILES
#undef DECLARE_XE20_MOE_MXFP8_EXTERN

#define LAUNCH_MOE_MXFP8(...)                 \
  Xe20MoEGEMMMxfp8W8A16Launcher<__VA_ARGS__>( \
      queue,                                  \
      activations.data_ptr(),                 \
      packed_weights.data_ptr(),              \
      scales.data_ptr(),                      \
      bias_ptr,                               \
      output.data_ptr(),                      \
      gemm_n,                                 \
      gemm_k,                                 \
      total_rows_for_experts.data_ptr<int>(), \
      n_experts,                              \
      atomic_buffer.data_ptr<int>(),          \
      1.0f,                                   \
      7.0f)

#define DISPATCH_MOE_MXFP8(WithBias, ...)          \
  do {                                             \
    if (WithBias) {                                \
      LAUNCH_MOE_MXFP8(__VA_ARGS__, 0, false, true);  \
    } else {                                       \
      LAUNCH_MOE_MXFP8(__VA_ARGS__, 0, false, false); \
    }                                              \
  } while (0)

// Dense/grouped MXFP8 W8A16 GEMM: output[M,N] = activations[M,K] @ B[N,K]^T
// with per-32-K-block fp32 (direct-multiplier) scales. Non-fused, N-outer.
// For a dense linear the caller passes n_experts == 1 and total_rows == [M].
void moe_grouped_mm_nt_xe20_mxfp8_w8a16(
    torch::Tensor& output,
    const torch::Tensor& activations,
    const torch::Tensor& packed_weights,  // [E, N, K] float8_e4m3fn (one byte/elem)
    const torch::Tensor& scales,          // [E, N, K/32] float32 direct multiplier (N-outer)
    const std::optional<at::Tensor>& bias,
    const torch::Tensor& total_rows_for_experts,
    const int64_t n_experts) {
  int total_m = activations.sizes()[0];
  int gemm_k = activations.sizes()[1];
  auto pw_shape = packed_weights.sizes().vec();
  int gemm_n = pw_shape[1];

  TORCH_CHECK(pw_shape.size() == 3, "packed_weights must be 3D [E, N, K]");
  TORCH_CHECK(pw_shape[0] == n_experts, "packed_weights first dim must equal n_experts");
  TORCH_CHECK(pw_shape[1] == gemm_n, "packed_weights second dim must equal N");
  TORCH_CHECK(pw_shape[2] == gemm_k, "packed_weights last dim must equal K (one e4m3 byte per element)");
  TORCH_CHECK(
      packed_weights.scalar_type() == at::ScalarType::Float8_e4m3fn, "packed_weights must be float8_e4m3fn");

  auto sc_shape = scales.sizes().vec();
  TORCH_CHECK(sc_shape.size() == 3, "scales must be 3D [E, N, K/32]");
  TORCH_CHECK(sc_shape[0] == n_experts, "scales first dim must equal n_experts");
  TORCH_CHECK(sc_shape[1] == gemm_n, "scales second dim must equal N");
  TORCH_CHECK(
      sc_shape[2] == gemm_k / MoE_MXFP8_W8A16::MXFP8_GROUP_SIZE, "scales last dim must equal K/32");
  TORCH_CHECK(scales.scalar_type() == at::ScalarType::Float, "scales must be float32 (direct multiplier)");

  TORCH_CHECK(
      n_experts == total_rows_for_experts.size(0),
      "total_rows_for_experts must have the same size as packed_weights.size(0)");
  TORCH_CHECK(output.sizes()[0] == total_m, "output rows must match activations rows");
  TORCH_CHECK(output.sizes()[1] == gemm_n, "output must have N columns");

  TORCH_CHECK(activations.scalar_type() == at::ScalarType::BFloat16, "activations must be bfloat16");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16, "output must be bfloat16");
  TORCH_CHECK(
      gemm_k % MoE_MXFP8_W8A16::MXFP8_GROUP_SIZE == 0, "K must be a multiple of GROUP_SIZE=32");

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
    DISPATCH_MOE_MXFP8(with_bias, Shape<_8, _64, _32>, Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>);
  } else if (avg_m <= 128 && small_weight) {
    DISPATCH_MOE_MXFP8(with_bias, Shape<_128, _128, _32>, Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>);
  } else {
    DISPATCH_MOE_MXFP8(with_bias, Shape<_256, _256, _32>, Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>);
  }
}

#undef SYCL_INTEL_TARGET
