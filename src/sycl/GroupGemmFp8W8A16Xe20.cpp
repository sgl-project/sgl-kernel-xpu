#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>
#include <map>

#include "Utils.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "kernels/moe/xe20/fp8/moe_kernel.hpp"
#include "sgl_kernel_export.h"
#ifdef USE_MOE_JIT
#include "jit/moe_jit.h"
#endif

using namespace cute;

template <typename Tile, typename SGLayout, bool WeightScaleBlocked>
__attribute__((visibility("default"))) void Xe20MoEGEMMFp8W8A16Launcher(
    sycl::queue q,
    const void* activations,
    const void* weights,
    const void* weight_scales,
    const void* bias,
    void* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* num_rows_per_expert_device,
    const int num_experts,
    int* workspace,
    int ld_b,
    int weight_scale_count,
    bool static_scheduler);

using Tile_16_64_32 = Shape<_16, _64, _32>;
using Tile_32_64_32 = Shape<_32, _64, _32>;
using Tile_64_64_32 = Shape<_64, _64, _32>;
using Tile_128_128_16 = Shape<_128, _128, _16>;

using SG_1_4_1 = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
using SG_2_4_1 = Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>;
using SG_4_2_1 = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;

#define DECLARE_XE20_MOE_FP8_W8A16_EXTERN(Tile, SGLayout, WeightScaleBlocked)           \
  extern template void Xe20MoEGEMMFp8W8A16Launcher<Tile, SGLayout, WeightScaleBlocked>( \
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
      int*,                                                                             \
      int,                                                                              \
      int,                                                                              \
      bool);

#define DECLARE_XE20_MOE_FP8_W8A16_ALL_SCALE_VARIANTS(Tile, SGLayout) \
  DECLARE_XE20_MOE_FP8_W8A16_EXTERN(Tile, SGLayout, false)            \
  DECLARE_XE20_MOE_FP8_W8A16_EXTERN(Tile, SGLayout, true)

DECLARE_XE20_MOE_FP8_W8A16_ALL_SCALE_VARIANTS(Tile_16_64_32, SG_1_4_1)
DECLARE_XE20_MOE_FP8_W8A16_ALL_SCALE_VARIANTS(Tile_32_64_32, SG_1_4_1)
DECLARE_XE20_MOE_FP8_W8A16_EXTERN(Tile_64_64_32, SG_2_4_1, false)
DECLARE_XE20_MOE_FP8_W8A16_ALL_SCALE_VARIANTS(Tile_128_128_16, SG_4_2_1)

#undef DECLARE_XE20_MOE_FP8_W8A16_ALL_SCALE_VARIANTS
#undef DECLARE_XE20_MOE_FP8_W8A16_EXTERN

#define LAUNCH_MOE_FP8_W8A16(WeightScaleBlocked, ...)           \
  Xe20MoEGEMMFp8W8A16Launcher<__VA_ARGS__, WeightScaleBlocked>( \
      queue,                                                    \
      activations.data_ptr(),                                   \
      weights.data_ptr(),                                       \
      weight_scales.data_ptr(),                                 \
      bias_ptr,                                                 \
      output.data_ptr(),                                        \
      gemm_n,                                                   \
      gemm_k,                                                   \
      total_rows_for_experts.data_ptr<int>(),                   \
      n_experts,                                                \
      atomic_buffer.data_ptr<int>(),                            \
      ld_b,                                                     \
      scale_count,                                              \
      static_scheduler)

#define DISPATCH_MOE_FP8_W8A16_BLOCK_TILES()                                       \
  do {                                                                             \
    if (avg_m <= 4) {                                                              \
      LAUNCH_MOE_FP8_W8A16(true, Tile_16_64_32, SG_1_4_1);                         \
    } else if (avg_m >= 1024 || (avg_m > 128 && gemm_k >= 512 && gemm_n >= 512)) { \
      LAUNCH_MOE_FP8_W8A16(true, Tile_128_128_16, SG_4_2_1);                       \
    } else {                                                                       \
      LAUNCH_MOE_FP8_W8A16(true, Tile_32_64_32, SG_1_4_1);                         \
    }                                                                              \
  } while (0)

#define DISPATCH_MOE_FP8_W8A16_SCALAR_TILES()                                         \
  do {                                                                                \
    if (avg_m <= 8) {                                                                 \
      LAUNCH_MOE_FP8_W8A16(false, Tile_16_64_32, SG_1_4_1);                           \
    } else if (scale_count == 2 && avg_m > 32 && avg_m <= 128 && gemm_k >= 2048) {    \
      LAUNCH_MOE_FP8_W8A16(false, Tile_64_64_32, SG_2_4_1);                           \
    } else if (avg_m <= 32 || (scale_count == 2 && gemm_k >= 4096 && avg_m <= 512)) { \
      LAUNCH_MOE_FP8_W8A16(false, Tile_32_64_32, SG_1_4_1);                           \
    } else {                                                                          \
      LAUNCH_MOE_FP8_W8A16(false, Tile_128_128_16, SG_4_2_1);                         \
    }                                                                                 \
  } while (0)

SGL_KERNEL_EXPORT void moe_grouped_mm_nt_xe20_fp8_w8a16(
    torch::Tensor& output,
    const torch::Tensor& activations,
    const torch::Tensor& weights,
    const torch::Tensor& weight_scales,
    const std::optional<at::Tensor>& bias,
    const torch::Tensor& total_rows_for_experts,
    const int64_t n_experts) {
  CHECK_INPUT(output);
  CHECK_INPUT(activations);
  CHECK_INPUT(weights);
  CHECK_INPUT(weight_scales);
  CHECK_INPUT(total_rows_for_experts);
  TORCH_CHECK(output.device() == activations.device(), "output must be on the same device as activations");
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
    TORCH_CHECK(bias_tensor.scalar_type() == at::kFloat, "bias must be float32");
    TORCH_CHECK(bias_tensor.dim() == 2, "bias must be 2D [E, N]");
  }
  TORCH_CHECK(activations.scalar_type() == at::ScalarType::BFloat16, "W8A16 activations must be bfloat16");
  TORCH_CHECK(weights.scalar_type() == at::ScalarType::Float8_e4m3fn, "W8A16 weights must be float8_e4m3fn");
  TORCH_CHECK(weight_scales.scalar_type() == at::kFloat, "W8A16 weight scales must be float32");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16, "W8A16 output must be bfloat16");
  TORCH_CHECK(
      weight_scales.dim() == 2 || weight_scales.dim() == 3,
      "W8A16 weight scales must be [E, 1]/[E, 2] or [E, N/128, K/128]");
  if (weight_scales.dim() == 2) {
    TORCH_CHECK(weight_scales.size(1) == 1 || weight_scales.size(1) == 2, "W8A16 scale count must be 1 or 2");
  }
  TORCH_CHECK(n_experts > 0 && n_experts % 8 == 0, "n_experts must be a positive multiple of 8");
  TORCH_CHECK(activations.dim() == 2, "W8A16 activations must be 2D [M_total, K]");
  TORCH_CHECK(weights.dim() == 3, "W8A16 weights must be 3D [E, N, K]");
  TORCH_CHECK(output.dim() == 2, "W8A16 output must be 2D [M_total, N]");
  TORCH_CHECK(weights.size(0) == n_experts, "weights expert dimension mismatch");
  TORCH_CHECK(weight_scales.size(0) == n_experts, "weight scales expert dimension mismatch");
  TORCH_CHECK(
      total_rows_for_experts.dim() == 1 && total_rows_for_experts.size(0) == n_experts, "rows_for_experts must be [E]");
  TORCH_CHECK(total_rows_for_experts.scalar_type() == at::ScalarType::Int, "rows_for_experts must be int32");
  TORCH_CHECK(weights.size(2) == activations.size(1), "W8A16 K dimension mismatch");
  TORCH_CHECK(
      activations.is_contiguous() && weights.is_contiguous() && weight_scales.is_contiguous(),
      "W8A16 tensors must be contiguous");
  TORCH_CHECK(weights.size(1) % 64 == 0 && weights.size(2) % 32 == 0, "W8A16 N must be divisible by 64 and K by 32");

  int total_m = static_cast<int>(activations.size(0));
  int gemm_k = static_cast<int>(activations.size(1));
  int gemm_n = static_cast<int>(weights.size(1));
  int avg_m = total_m / static_cast<int>(n_experts);
  int ld_b = static_cast<int>(weights.stride(1));
  int scale_count = weight_scales.dim() == 2 ? static_cast<int>(weight_scales.size(1)) : 3;
  bool weight_scale_blocked = weight_scales.dim() == 3;
  bool static_scheduler = total_m <= n_experts || (!weight_scale_blocked && gemm_k <= 128);
  if (weight_scale_blocked) {
    TORCH_CHECK(weight_scales.size(1) == (gemm_n + 127) / 128, "W8A16 block scale N dimension must be ceil(N/128)");
    TORCH_CHECK(
        gemm_k % 128 == 0 && weight_scales.size(2) == gemm_k / 128, "W8A16 block scale K dimension must be K/128");
  }
  TORCH_CHECK(output.size(0) == total_m, "output rows must equal M_total");
  TORCH_CHECK(output.size(1) == gemm_n, "output must have the same columns as weights");
  if (bias.has_value()) {
    TORCH_CHECK(bias->size(0) == n_experts && bias->size(1) == gemm_n, "bias shape must be [E, N]");
  }
  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  using StreamKey = std::pair<c10::DeviceIndex, c10::StreamId>;
  thread_local std::map<StreamKey, at::Tensor> atomic_buffers;
  auto [buffer_it, inserted] = atomic_buffers.try_emplace(StreamKey{stream.device_index(), stream.id()}, at::Tensor{});
  at::Tensor& atomic_buffer = buffer_it->second;
  if (inserted || !atomic_buffer.defined()) {
    atomic_buffer = at::empty({1}, activations.options().dtype(at::kInt));
  }
  if (!static_scheduler) {
    queue.memset(atomic_buffer.data_ptr(), 0, sizeof(int32_t));
  }
  bool with_bias = bias.has_value();
  void* bias_ptr = with_bias ? bias->data_ptr() : nullptr;

#ifdef USE_MOE_JIT
  std::string jit_err;
  TORCH_CHECK(
      sgl::moe_jit::fp8_w8a16_grouped_gemm_launch(
          avg_m,
          scale_count,
          &queue,
          activations.data_ptr(),
          weights.data_ptr(),
          weight_scales.data_ptr(),
          bias_ptr,
          output.data_ptr(),
          gemm_n,
          gemm_k,
          total_rows_for_experts.data_ptr<int>(),
          static_cast<int>(n_experts),
          atomic_buffer.data_ptr<int>(),
          ld_b,
          static_scheduler,
          jit_arch_code(),
          &jit_err),
      jit_err);
#else
  if (scale_count == 3) {
    DISPATCH_MOE_FP8_W8A16_BLOCK_TILES();
  } else {
    DISPATCH_MOE_FP8_W8A16_SCALAR_TILES();
  }
#endif
}

#undef DISPATCH_MOE_FP8_W8A16_BLOCK_TILES
#undef DISPATCH_MOE_FP8_W8A16_SCALAR_TILES
#undef LAUNCH_MOE_FP8_W8A16

#undef SYCL_INTEL_TARGET
