#undef SYCL_INTEL_TARGET
#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cute/tensor.hpp>
#include <map>

#include "SGLKernelPerf.h"
#include "Utils.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "kernels/moe/xe20/w8a16/moe_kernel.hpp"
#include "sgl_kernel_export.h"
#ifdef USE_MOE_JIT
#include "jit/moe_jit.h"
#endif

using namespace cute;

template <typename Tile, typename SGLayout, bool IsInt8>
__attribute__((visibility("default"))) void Xe20MoEGEMMW8A16Launcher(
    sycl::queue q,
    const void* activations,
    const void* weights,
    const void* weight_scales,
    const void* weight_zeros,
    const void* bias,
    void* outputs,
    const int gemm_n,
    const int gemm_k,
    const int* num_rows_per_expert_device,
    const int num_experts,
    int* workspace,
    int ld_b,
    int64_t expert_stride_b,
    int64_t expert_stride_s,
    int scale_n_stride,
    int scale_k_stride,
    int64_t expert_stride_z,
    int zero_n_stride,
    int zero_k_stride,
    int scale_mode,
    int group_size,
    int n_block_shift,
    bool scale_is_bf16,
    bool is_signed_int8,
    bool static_scheduler);

using ScaleMode = moe_w8a16::ScaleMode;

using Tile_16_64_32 = Shape<_16, _64, _32>;
using Tile_32_64_32 = Shape<_32, _64, _32>;
using Tile_64_64_32 = Shape<_64, _64, _32>;
using Tile_128_128_32 = Shape<_128, _128, _32>;

using SG_1_4_1 = Layout<Shape<_1, _4, _1>, Stride<_4, _1, _0>>;
using SG_2_4_1 = Layout<Shape<_2, _4, _1>, Stride<_4, _1, _0>>;
using SG_4_4_1 = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;

#define DECLARE_XE20_MOE_W8A16_EXTERN(Tile, SGLayout, IsInt8)            \
  extern template void Xe20MoEGEMMW8A16Launcher<Tile, SGLayout, IsInt8>( \
      sycl::queue,                                                       \
      const void*,                                                       \
      const void*,                                                       \
      const void*,                                                       \
      const void*,                                                       \
      const void*,                                                       \
      void*,                                                             \
      const int,                                                         \
      const int,                                                         \
      const int*,                                                        \
      const int,                                                         \
      int*,                                                              \
      int,                                                               \
      int64_t,                                                           \
      int64_t,                                                           \
      int,                                                               \
      int,                                                               \
      int64_t,                                                           \
      int,                                                               \
      int,                                                               \
      int,                                                               \
      int,                                                               \
      int,                                                               \
      bool,                                                              \
      bool,                                                              \
      bool);

#define DECLARE_XE20_MOE_W8A16_ALL_WEIGHT_KINDS(Tile, SGLayout) \
  DECLARE_XE20_MOE_W8A16_EXTERN(Tile, SGLayout, false)          \
  DECLARE_XE20_MOE_W8A16_EXTERN(Tile, SGLayout, true)

DECLARE_XE20_MOE_W8A16_ALL_WEIGHT_KINDS(Tile_16_64_32, SG_1_4_1)
DECLARE_XE20_MOE_W8A16_ALL_WEIGHT_KINDS(Tile_32_64_32, SG_1_4_1)
DECLARE_XE20_MOE_W8A16_ALL_WEIGHT_KINDS(Tile_64_64_32, SG_2_4_1)
DECLARE_XE20_MOE_W8A16_ALL_WEIGHT_KINDS(Tile_128_128_32, SG_4_4_1)

#undef DECLARE_XE20_MOE_W8A16_ALL_WEIGHT_KINDS
#undef DECLARE_XE20_MOE_W8A16_EXTERN

#define LAUNCH_MOE_W8A16_KIND(IsInt8, Tile, SGLayout) \
  Xe20MoEGEMMW8A16Launcher<Tile, SGLayout, IsInt8>(   \
      queue,                                          \
      activations.data_ptr(),                         \
      weights_b.data_ptr(),                           \
      weight_scales.data_ptr(),                       \
      zero_ptr,                                       \
      bias_ptr,                                       \
      output.data_ptr(),                              \
      gemm_n,                                         \
      gemm_k,                                         \
      total_rows_for_experts.data_ptr<int>(),         \
      static_cast<int>(n_experts),                    \
      atomic_buffer.data_ptr<int>(),                  \
      ld_b,                                           \
      expert_stride_b,                                \
      expert_stride_s,                                \
      scale_n_stride,                                 \
      scale_k_stride,                                 \
      expert_stride_z,                                \
      zero_n_stride,                                  \
      zero_k_stride,                                  \
      scale_mode,                                     \
      group_size,                                     \
      n_block_shift,                                  \
      scale_is_bf16,                                  \
      is_signed_int8,                                 \
      static_scheduler)

#define LAUNCH_MOE_W8A16(Tile, SGLayout)            \
  do {                                              \
    if (is_int8) {                                  \
      LAUNCH_MOE_W8A16_KIND(true, Tile, SGLayout);  \
    } else {                                        \
      LAUNCH_MOE_W8A16_KIND(false, Tile, SGLayout); \
    }                                               \
  } while (0)

SGL_KERNEL_EXPORT void moe_grouped_mm_nt_xe20_w8a16(
    torch::Tensor& output,
    const torch::Tensor& activations,
    const torch::Tensor& weights,
    const torch::Tensor& weight_scales,
    const std::optional<at::Tensor>& bias,
    const torch::Tensor& total_rows_for_experts,
    const int64_t n_experts,
    const std::optional<at::Tensor>& weight_zeros) {
  CHECK_INPUT(2, at::ScalarType::BFloat16, output);
  CHECK_INPUT(2, at::ScalarType::BFloat16, activations);
  TORCH_CHECK(weights.device().is_xpu(), "weights must be an XPU tensor");
  TORCH_CHECK(weights.dim() == 3, "W8A16 weights must be 3D");
  const auto w_dtype = weights.scalar_type();
  const bool is_fp8 = (w_dtype == at::ScalarType::Float8_e4m3fn);
  const bool is_int8 = (w_dtype == at::ScalarType::Char || w_dtype == at::ScalarType::Byte);
  const bool is_signed_int8 = (w_dtype == at::ScalarType::Char);
  TORCH_CHECK(is_fp8 || is_int8, "W8A16 weights must be float8_e4m3fn, int8, or uint8");

  CHECK_DEVICE(weight_scales);
  CHECK_INPUT(1, at::ScalarType::Int, total_rows_for_experts);
  TORCH_CHECK(output.device() == activations.device(), "output must be on the same device as activations");
  TORCH_CHECK(weights.device() == activations.device(), "weights must be on the same device as activations");
  TORCH_CHECK(
      weight_scales.device() == activations.device(), "weight_scales must be on the same device as activations");
  TORCH_CHECK(
      total_rows_for_experts.device() == activations.device(),
      "total_rows_for_experts must be on the same device as activations");
  if (bias.has_value()) {
    const auto& bias_tensor = *bias;
    CHECK_INPUT(2, at::kFloat, bias_tensor);
    TORCH_CHECK(bias_tensor.device() == activations.device(), "bias must be on the same device as activations");
  }

  TORCH_CHECK(n_experts > 0, "n_experts must be positive");
  CHECK_SHAPE(total_rows_for_experts, n_experts);

  const int total_m = static_cast<int>(activations.size(0));
  const int gemm_k = static_cast<int>(activations.size(1));
  const int gemm_n = static_cast<int>(output.size(1));
  TORCH_CHECK(output.size(0) == total_m, "output rows must equal M_total");
  TORCH_CHECK(weights.size(0) == n_experts, "weights expert dimension mismatch");
  TORCH_CHECK(weight_scales.size(0) == n_experts, "weight scales expert dimension mismatch");
  TORCH_CHECK(
      gemm_n >= 64 && gemm_n % 16 == 0 && gemm_k % 32 == 0,
      "W8A16 requires N >= 64 divisible by 16 and K divisible by 32");

  // Resolve Layout B:
  //   1) Preferred N-major view `[E, N, K]` with `stride(1) == 1` (zero-copy).
  //   2) Non-square physical `[E, K, N]` with `stride(2) == 1` (zero-copy view).
  //   3) Legacy K-major `[E, N, K]` with `stride(2) == 1` (converted on stream).
  at::Tensor weights_b;
  if (weights.size(1) == gemm_n && weights.size(2) == gemm_k && weights.stride(1) == 1 && weights.stride(2) >= gemm_n &&
      weights.stride(0) >= static_cast<int64_t>(gemm_k) * weights.stride(2)) {
    weights_b = weights;
  } else if (
      gemm_n != gemm_k && weights.size(1) == gemm_k && weights.size(2) == gemm_n && weights.stride(2) == 1 &&
      weights.stride(1) >= gemm_n && weights.stride(0) >= static_cast<int64_t>(gemm_k) * weights.stride(1)) {
    weights_b = weights.transpose(1, 2);
  } else {
    TORCH_CHECK(
        weights.size(1) == gemm_n && weights.size(2) == gemm_k && weights.stride(2) == 1,
        "W8A16 weights shape/stride mismatch: expected [E=",
        n_experts,
        ", N=",
        gemm_n,
        ", K=",
        gemm_k,
        "] with unit N-stride or unit K-stride, got shape ",
        weights.sizes(),
        " and stride ",
        weights.strides());
    weights_b = weights.transpose(1, 2).contiguous().transpose(1, 2);
  }

  const int ld_b = static_cast<int>(weights_b.stride(2));
  const int64_t expert_stride_b = weights_b.stride(0);
  TORCH_CHECK(
      (ld_b % 16) == 0 && (expert_stride_b % 64) == 0,
      "W8A16 Layout-B requires 16-byte pitch alignment (ld_b % 16 == 0) and "
      "64-byte expert base alignment (expert_stride_b % 64 == 0), got ld_b=",
      ld_b,
      ", expert_stride_b=",
      expert_stride_b);

  const bool is_mxfp8 = !is_int8 && (weight_scales.scalar_type() == at::ScalarType::Byte ||
                                     weight_scales.scalar_type() == at::ScalarType::Float8_e8m0fnu);
  bool scale_is_bf16 = false;
  int scale_mode = static_cast<int>(ScaleMode::ScalarSingle);
  int group_size = gemm_k;
  int n_block_shift = 0;

  if (is_mxfp8) {
    CHECK_DIM(3, weight_scales);
    TORCH_CHECK(weight_scales.size(1) == gemm_n, "MXFP8 scale N dimension must equal N");
    TORCH_CHECK(gemm_k % 32 == 0 && weight_scales.size(2) == gemm_k / 32, "MXFP8 scale K dimension must be K/32");
    scale_mode = static_cast<int>(ScaleMode::BlockMXFP8);
    group_size = 32;
  } else {
    TORCH_CHECK(
        weight_scales.scalar_type() == at::kFloat || weight_scales.scalar_type() == at::kBFloat16,
        "W8A16 non-MXFP8 weight_scales must be float32 or bfloat16");
    scale_is_bf16 = (weight_scales.scalar_type() == at::kBFloat16);
    if (weight_scales.dim() == 1) {
      scale_mode = static_cast<int>(ScaleMode::ScalarSingle);
      group_size = gemm_k;
    } else if (weight_scales.dim() == 2) {
      if (weight_scales.size(1) == 1) {
        scale_mode = static_cast<int>(ScaleMode::ScalarSingle);
      } else if (weight_scales.size(1) == 2) {
        scale_mode = static_cast<int>(ScaleMode::ScalarGateUp);
      } else if (weight_scales.size(1) == gemm_n) {
        scale_mode = static_cast<int>(ScaleMode::PerChannelN);
      } else {
        TORCH_CHECK(
            false, "W8A16 2D weight_scales second dim must be 1, 2, or N (", gemm_n, "), got ", weight_scales.size(1));
      }
      group_size = gemm_k;
    } else if (weight_scales.dim() == 3) {
      const int64_t s_n = weight_scales.size(1);
      const int64_t s_k = weight_scales.size(2);
      TORCH_CHECK(
          s_n >= 1 && s_n <= gemm_n, "W8A16 3D weight_scales size(1) must be in [1, N (", gemm_n, ")], got ", s_n);
      TORCH_CHECK(
          s_k >= 1 && gemm_k % s_k == 0,
          "W8A16 K (",
          gemm_k,
          ") must be divisible by weight_scales.size(2) (",
          s_k,
          ")");
      if (s_n == gemm_n && s_k == 1) {
        scale_mode = static_cast<int>(ScaleMode::PerChannelN);
        group_size = gemm_k;
      } else {
        group_size = static_cast<int>(gemm_k / s_k);
        if (s_k > 1) {
          TORCH_CHECK(
              group_size >= 32 && (group_size & (group_size - 1)) == 0,
              "W8A16 group_size must be a power of two >= 32, got ",
              group_size);
        }
        if (s_n == gemm_n) {
          scale_mode = static_cast<int>(ScaleMode::GroupNK);
          n_block_shift = 0;
        } else if (s_n == 1) {
          scale_mode = static_cast<int>(ScaleMode::BlockFP32);
          n_block_shift = 30;
        } else {
          int matched_shift = -1;
          for (int sh = 1; sh <= 20; ++sh) {
            const int64_t blk_n = int64_t{1} << sh;
            if (((gemm_n + blk_n - 1) >> sh) == s_n) {
              matched_shift = sh;
              break;
            }
          }
          TORCH_CHECK(
              matched_shift > 0,
              "Unsupported 3D W8A16 weight_scales shape ",
              weight_scales.sizes(),
              " for N=",
              gemm_n,
              ", K=",
              gemm_k,
              " (expected size(1) == ceil(N / block_n) for a power-of-two block_n)");
          scale_mode = static_cast<int>(ScaleMode::BlockFP32);
          n_block_shift = matched_shift;
        }
      }
    } else {
      TORCH_CHECK(false, "W8A16 weight_scales must be 1D, 2D, or 3D");
    }
  }

  auto resolve_scale_like_strides =
      [](const torch::Tensor& t, const char* name, int64_t& expert_stride, int& n_stride, int& k_stride) {
        if (t.dim() <= 2) {
          TORCH_CHECK(t.is_contiguous(), "W8A16 1D/2D ", name, " must be contiguous");
          expert_stride = t.stride(0);
          n_stride = 1;
          k_stride = 1;
          return;
        }
        if (t.is_contiguous()) {
          expert_stride = t.stride(0);
          n_stride = static_cast<int>(t.stride(1));
          k_stride = static_cast<int>(t.stride(2));
          return;
        }
        TORCH_CHECK(
            t.stride(1) == 1 && t.stride(2) >= t.size(1) && t.stride(0) >= t.size(2) * t.stride(2),
            "W8A16 3D ",
            name,
            " must be K-major contiguous or N-major (stride(1) == 1), got shape ",
            t.sizes(),
            " and stride ",
            t.strides());
        expert_stride = t.stride(0);
        n_stride = 1;
        k_stride = static_cast<int>(t.stride(2));
      };

  int64_t expert_stride_s = 1;
  int scale_n_stride = 1;
  int scale_k_stride = 1;
  resolve_scale_like_strides(weight_scales, "weight_scales", expert_stride_s, scale_n_stride, scale_k_stride);

  const void* zero_ptr = nullptr;
  int64_t expert_stride_z = 0;
  int zero_n_stride = 1;
  int zero_k_stride = 1;
  if (weight_zeros.has_value()) {
    const auto& zp = *weight_zeros;
    TORCH_CHECK(is_int8 && !is_signed_int8, "W8A16 weight_zeros is only supported for unsigned uint8 weights");
    CHECK_DEVICE(zp);
    TORCH_CHECK(zp.device() == activations.device(), "weight_zeros must be on the same device as activations");
    TORCH_CHECK(zp.scalar_type() == at::ScalarType::Byte, "W8A16 weight_zeros must be uint8");
    TORCH_CHECK(zp.size(0) == n_experts, "weight_zeros expert dimension mismatch");
    if (scale_mode == static_cast<int>(ScaleMode::PerChannelN)) {
      TORCH_CHECK(
          (zp.dim() == 2 && zp.size(1) == gemm_n) || (zp.dim() == 3 && zp.size(1) == gemm_n && zp.size(2) == 1),
          "Per-channel weight_zeros must have shape [E, N] or [E, N, 1]");
    } else if (
        scale_mode == static_cast<int>(ScaleMode::GroupNK) || scale_mode == static_cast<int>(ScaleMode::BlockFP32)) {
      TORCH_CHECK(
          zp.dim() == 3 && zp.sizes() == weight_scales.sizes(),
          "Block/groupwise weight_zeros must match weight_scales shape ",
          weight_scales.sizes());
    } else {
      TORCH_CHECK(false, "W8A16 weight_zeros requires per-channel [E, N] or 3D block/group scales");
    }
    resolve_scale_like_strides(zp, "weight_zeros", expert_stride_z, zero_n_stride, zero_k_stride);
    zero_ptr = zp.data_ptr();
  }

  if (bias.has_value()) {
    TORCH_CHECK(bias->size(0) == n_experts && bias->size(1) == gemm_n, "bias shape must be [E, N]");
  }

  const int avg_m = static_cast<int>(total_m / n_experts);
  const bool weight_scale_blocked = moe_w8a16::is_block_scale_mode(scale_mode) || (zero_ptr != nullptr);
  const int64_t total_weight_bytes = static_cast<int64_t>(n_experts) * gemm_n * gemm_k;
  // Static round-robin scheduling avoids atomic setup on tiny matrices, whereas
  // dynamic work-stealing keeps workgroups packed onto 1-2 experts at a time to
  // avoid DTLB thrashing when total expert weights exceed L2/DTLB reach.
  const bool static_scheduler =
      (!weight_scale_blocked && gemm_k <= 128) || (total_m <= n_experts && total_weight_bytes <= (4LL << 20));

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
  void* bias_ptr = bias.has_value() ? bias->data_ptr() : nullptr;

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  const double flops = 2.0 * static_cast<double>(total_m) * static_cast<double>(gemm_n) * static_cast<double>(gemm_k);
  const double a_bytes =
      static_cast<double>(total_m) * static_cast<double>(gemm_k) * static_cast<double>(activations.element_size());
  const double w_bytes = static_cast<double>(n_experts) * static_cast<double>(gemm_n) * static_cast<double>(gemm_k) *
                         static_cast<double>(weights_b.element_size());
  const double out_bytes =
      static_cast<double>(total_m) * static_cast<double>(gemm_n) * static_cast<double>(output.element_size());
  const double bytes = a_bytes + w_bytes + out_bytes;
  SGL_KERNEL_PERF_SCOPE("moe_grouped_mm_nt_xe20_w8a16", queue, bytes, flops);
#endif

#ifdef USE_MOE_JIT
  std::string jit_err;
  TORCH_CHECK(
      sgl::moe_jit::w8a16_grouped_gemm_launch(
          avg_m,
          scale_mode,
          is_int8,
          &queue,
          activations.data_ptr(),
          weights_b.data_ptr(),
          weight_scales.data_ptr(),
          zero_ptr,
          bias_ptr,
          output.data_ptr(),
          gemm_n,
          gemm_k,
          total_rows_for_experts.data_ptr<int>(),
          static_cast<int>(n_experts),
          atomic_buffer.data_ptr<int>(),
          ld_b,
          expert_stride_b,
          expert_stride_s,
          scale_n_stride,
          scale_k_stride,
          expert_stride_z,
          zero_n_stride,
          zero_k_stride,
          group_size,
          n_block_shift,
          scale_is_bf16,
          is_signed_int8,
          static_scheduler,
          jit_arch_code(),
          &jit_err),
      jit_err);
#else
  const int tile_id = moe_w8a16::w8a16_select_tile(avg_m, gemm_n);
  switch (tile_id) {
    case 0:
      LAUNCH_MOE_W8A16(Tile_16_64_32, SG_1_4_1);
      break;
    case 1:
      LAUNCH_MOE_W8A16(Tile_32_64_32, SG_1_4_1);
      break;
    case 2:
      LAUNCH_MOE_W8A16(Tile_64_64_32, SG_2_4_1);
      break;
    default:
      LAUNCH_MOE_W8A16(Tile_128_128_32, SG_4_4_1);
      break;
  }
#endif
}

#undef LAUNCH_MOE_W8A16
#undef LAUNCH_MOE_W8A16_KIND

#undef SYCL_INTEL_TARGET
