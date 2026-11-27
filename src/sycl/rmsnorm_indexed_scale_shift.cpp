#include <ATen/ATen.h>
#include <ATen/ops/rms_norm.h>
#include <ATen/xpu/XPUContext.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "sgl_kernel_export.h"

namespace at::native::xpu {
namespace {
using bf16 = sycl::ext::oneapi::bfloat16;
struct alignas(8) Bf16x4 {
  bf16 values[4];
};

inline bf16 round_output(float value) {
  // Match the NaN encoding of eager XPU float-to-BF16 conversions.
  return sycl::isnan(value) ? sycl::bit_cast<bf16>(uint16_t{0xffc1}) : static_cast<bf16>(value);
}
constexpr int kVector = 4;
constexpr int kSubgroup = 32;
constexpr int kModulationThreads = 128;
template <typename Index>
class RMSNormIndexedScaleShiftKernel;
template <typename Index>
class IndexedScaleShiftKernel;

template <typename Index>
void launch_rmsnorm_indexed_scale_shift(
    sycl::queue& queue,
    const bf16* x,
    const bf16* weight,
    const bf16* shift,
    const bf16* scale,
    const Index* indices,
    bf16* output,
    int64_t rows,
    int64_t hidden,
    int64_t shift_stride,
    int64_t scale_stride,
    int64_t index_stride,
    float eps,
    int threads,
    int physical_threads) {
  // Keep the reference's logical lanes and addition order while reducing the
  // workgroup size for batches large enough to occupy the GPU.
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> partials(sycl::range<1>(threads / kSubgroup), cgh);
    cgh.parallel_for<RMSNormIndexedScaleShiftKernel<Index>>(
        sycl::nd_range<1>(sycl::range<1>(rows * physical_threads), sycl::range<1>(physical_threads)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSubgroup)]] {
          const int64_t row = item.get_group(0);
          const int64_t lane = item.get_local_id(0);
          const bf16* x_row = x + row * hidden;
          auto sg = item.get_sub_group();
          const int group = sg.get_group_linear_id();
          const int sublane = sg.get_local_linear_id();
          const int groups = threads / kSubgroup;
          // PyTorch XPU accumulates four adjacent columns per logical lane.
          for (int logical_lane = lane; logical_lane < threads; logical_lane += physical_threads) {
            float squares = 0.0f;
            for (int64_t base = logical_lane * kVector; base < hidden; base += threads * kVector) {
              const Bf16x4 values = reinterpret_cast<const Bf16x4*>(x_row)[base / kVector];
#pragma unroll
              for (int j = 0; j < kVector; ++j) {
                const float value = static_cast<float>(values.values[j]);
                squares += value * value;
              }
            }
            // Descending shuffles preserve the reference reduction tree.
            for (int offset = kSubgroup / 2; offset > 0; offset /= 2) {
              squares += sycl::shift_group_left(sg, squares, offset);
            }
            if (sublane == 0) partials[logical_lane / kSubgroup] = squares;
          }
          sycl::group_barrier(item.get_group());
          if (groups > 1) {
            if (group == 0) {
              float total = sublane < groups ? partials[sublane] : 0.0f;
              for (int offset = groups / 2; offset > 0; offset /= 2) {
                total += sycl::shift_group_left(sg, total, offset);
              }
              if (sublane == 0) partials[0] = total;
            }
            sycl::group_barrier(item.get_group());
          }
          const float inv_rms = sycl::rsqrt(partials[0] / static_cast<float>(hidden) + eps);
          const int64_t table_row = static_cast<int64_t>(indices[row * index_stride]);
          const bf16* shift_row = shift + table_row * shift_stride;
          const bf16* scale_row = scale + table_row * scale_stride;
          bf16* out_row = output + row * hidden;
          for (int64_t base = lane * kVector; base < hidden; base += physical_threads * kVector) {
            const Bf16x4 values = reinterpret_cast<const Bf16x4*>(x_row)[base / kVector];
            const Bf16x4 weights = reinterpret_cast<const Bf16x4*>(weight)[base / kVector];
            Bf16x4 out;
#pragma unroll
            for (int j = 0; j < kVector; ++j) {
              const int64_t col = base + j;
              const bf16 normalized = static_cast<bf16>(
                  static_cast<float>(weights.values[j]) * (inv_rms * static_cast<float>(values.values[j])));
              const bf16 factor = static_cast<bf16>(1.0f + static_cast<float>(scale_row[col]));
              const bf16 scaled = static_cast<bf16>(static_cast<float>(normalized) * static_cast<float>(factor));
              out.values[j] = round_output(static_cast<float>(scaled) + static_cast<float>(shift_row[col]));
            }
            reinterpret_cast<Bf16x4*>(out_row)[base / kVector] = out;
          }
        });
  });
}

// Non-vectorizable rows/weights take PyTorch's scalar RMSNorm path. Reuse that
// calculation and fuse only the modulation, preserving the supported layouts.
template <typename Index>
void launch_indexed_scale_shift(
    sycl::queue& queue,
    const bf16* normalized,
    const bf16* shift,
    const bf16* scale,
    const Index* indices,
    bf16* output,
    int64_t rows,
    int64_t hidden,
    int64_t shift_stride,
    int64_t scale_stride,
    int64_t index_stride) {
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<IndexedScaleShiftKernel<Index>>(
        sycl::nd_range<1>(sycl::range<1>(rows * kModulationThreads), sycl::range<1>(kModulationThreads)),
        [=](sycl::nd_item<1> item) {
          const int64_t row = item.get_group(0);
          const int64_t table_row = static_cast<int64_t>(indices[row * index_stride]);
          for (int64_t col = item.get_local_id(0); col < hidden; col += kModulationThreads) {
            const bf16 factor = static_cast<bf16>(1.0f + static_cast<float>(scale[table_row * scale_stride + col]));
            const bf16 scaled =
                static_cast<bf16>(static_cast<float>(normalized[row * hidden + col]) * static_cast<float>(factor));
            output[row * hidden + col] =
                round_output(static_cast<float>(scaled) + static_cast<float>(shift[table_row * shift_stride + col]));
          }
        });
  });
}
}  // namespace

// Indices must be stride-one int32/int64 values in [0, shift.size(0)) for
// non-empty input. Debug builds validate values; Release builds trust them.
// Returns a new tensor; x, weight, shift, and scale are never modified.
SGL_KERNEL_EXPORT at::Tensor rmsnorm_indexed_scale_shift(
    const at::Tensor& x,
    const at::Tensor& weight,
    const at::Tensor& shift,
    const at::Tensor& scale,
    const at::Tensor& indices,
    double eps) {
  TORCH_CHECK(
      x.is_xpu() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(),
      "indexed RMSNorm/scale-shift requires contiguous BF16 XPU rows");
  const auto rows = x.size(0);
  const auto hidden = x.size(1);
  TORCH_CHECK(hidden > 0, "indexed RMSNorm/scale-shift hidden size must be positive");
  TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "indexed RMSNorm/scale-shift eps must be positive and finite");
  TORCH_CHECK(
      weight.device() == x.device() && weight.scalar_type() == x.scalar_type() && weight.dim() == 1 &&
          weight.numel() == hidden && weight.is_contiguous(),
      "indexed RMSNorm/scale-shift weight mismatch");
  TORCH_CHECK(
      shift.device() == x.device() && scale.device() == x.device() && shift.scalar_type() == x.scalar_type() &&
          scale.scalar_type() == x.scalar_type() && shift.dim() == 2 && scale.dim() == 2 &&
          shift.sizes() == scale.sizes() && (rows == 0 || shift.size(0) > 0) && shift.size(1) == hidden &&
          shift.stride(1) == 1 && scale.stride(1) == 1,
      "indexed RMSNorm/scale-shift modulation table mismatch");
  TORCH_CHECK(
      indices.device() == x.device() && indices.dim() == 1 && indices.numel() == rows &&
          (indices.scalar_type() == at::kLong || indices.scalar_type() == at::kInt) && indices.stride(0) == 1,
      "indexed RMSNorm/scale-shift indices mismatch");
  c10::DeviceGuard guard(x.device());
  auto output = at::empty_like(x);
  if (rows == 0) return output;
#ifdef SGL_KERNEL_DEBUG_CHECKS
  // A device-to-host sync is acceptable only in the debug-checks build.
  auto [min_index, max_index] = torch::aminmax(indices);
  TORCH_CHECK(
      min_index.item<int64_t>() >= 0 && max_index.item<int64_t>() < shift.size(0),
      "indexed RMSNorm/scale-shift indices must be in [0, ",
      shift.size(0),
      ")");
#endif
  auto& queue = at::xpu::getCurrentXPUStream().queue();
  auto* x_ptr = reinterpret_cast<const bf16*>(x.data_ptr<at::BFloat16>());
  auto* w_ptr = reinterpret_cast<const bf16*>(weight.data_ptr<at::BFloat16>());
  auto* shift_ptr = reinterpret_cast<const bf16*>(shift.data_ptr<at::BFloat16>());
  auto* scale_ptr = reinterpret_cast<const bf16*>(scale.data_ptr<at::BFloat16>());
  auto* out_ptr = reinterpret_cast<bf16*>(output.data_ptr<at::BFloat16>());
  auto* prop = at::xpu::getDeviceProperties(x.get_device());
  const bool vectorizable =
      hidden <= 8192 && hidden % kVector == 0 && reinterpret_cast<uintptr_t>(x_ptr) % alignof(Bf16x4) == 0 &&
      reinterpret_cast<uintptr_t>(w_ptr) % alignof(Bf16x4) == 0 && prop->max_work_group_size <= kSubgroup * kSubgroup;
  if (!vectorizable) {
    auto normalized = at::rms_norm(x, {hidden}, weight, eps);
    const auto* normalized_ptr = reinterpret_cast<const bf16*>(normalized.data_ptr<at::BFloat16>());
    if (indices.scalar_type() == at::kLong) {
      launch_indexed_scale_shift(
          queue,
          normalized_ptr,
          shift_ptr,
          scale_ptr,
          indices.data_ptr<int64_t>(),
          out_ptr,
          rows,
          hidden,
          shift.stride(0),
          scale.stride(0),
          indices.stride(0));
    } else {
      launch_indexed_scale_shift(
          queue,
          normalized_ptr,
          shift_ptr,
          scale_ptr,
          indices.data_ptr<int32_t>(),
          out_ptr,
          rows,
          hidden,
          shift.stride(0),
          scale.stride(0),
          indices.stride(0));
    }
    return output;
  }
  // Match PyTorch XPU's vectorized RMSNorm workgroup selection.
  int threads = prop->max_work_group_size;
  if (hidden / kVector <= threads) {
    while (threads > hidden / kVector && threads > kSubgroup)
      threads /= 2;
    if (threads > 128 && rows * 4 > prop->gpu_eu_count * prop->gpu_hw_threads_per_eu / 2) threads = 128;
  }
  const int physical_threads = rows < prop->gpu_eu_count ? threads : std::min(threads, 128);
  if (indices.scalar_type() == at::kLong) {
    launch_rmsnorm_indexed_scale_shift(
        queue,
        x_ptr,
        w_ptr,
        shift_ptr,
        scale_ptr,
        indices.data_ptr<int64_t>(),
        out_ptr,
        rows,
        hidden,
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        static_cast<float>(eps),
        threads,
        physical_threads);
  } else {
    launch_rmsnorm_indexed_scale_shift(
        queue,
        x_ptr,
        w_ptr,
        shift_ptr,
        scale_ptr,
        indices.data_ptr<int32_t>(),
        out_ptr,
        rows,
        hidden,
        shift.stride(0),
        scale.stride(0),
        indices.stride(0),
        static_cast<float>(eps),
        threads,
        physical_threads);
  }
  return output;
}
}  // namespace at::native::xpu
