#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <sycl/sycl.hpp>

#include "MemoryAccess.h"
#include "sgl_kernel_export.h"

namespace at::native::xpu {
namespace {
using bf16 = sycl::ext::oneapi::bfloat16;
constexpr int kScalarThreads = 256;
// A 5376-wide H3 row has 672 aligned 8-element vectors.
constexpr int kVectorThreads = 672;
constexpr int kVectorWidth = 8;
template <typename Index>
class H3IndexedGateKernel;
template <typename Index>
class H3IndexedGateVectorKernel;

template <typename Index>
void launch_h3_indexed_gate(
    sycl::queue& queue,
    const bf16* x,
    const bf16* gate,
    const bf16* other,
    const Index* indices,
    bf16* out,
    int64_t rows,
    int64_t hidden,
    int64_t gate_stride,
    int64_t index_stride) {
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<H3IndexedGateKernel<Index>>(
        sycl::nd_range<1>(sycl::range<1>(rows * kScalarThreads), sycl::range<1>(kScalarThreads)),
        [=](sycl::nd_item<1> item) {
          const int64_t row = item.get_group(0);
          const int64_t col0 = item.get_local_id(0);
          const int64_t gate_row = static_cast<int64_t>(indices[row * index_stride]);
          for (int64_t col = col0; col < hidden; col += kScalarThreads) {
            const int64_t offset = row * hidden + col;
            const bf16 product = static_cast<bf16>(
                static_cast<float>(gate[gate_row * gate_stride + col]) * static_cast<float>(other[offset]));
            out[offset] = static_cast<bf16>(static_cast<float>(x[offset]) + static_cast<float>(product));
          }
        });
  });
}

template <typename Index>
void launch_h3_indexed_gate_vector(
    sycl::queue& queue,
    const bf16* gate,
    const bf16* other,
    const Index* indices,
    bf16* out,
    int64_t rows,
    int64_t hidden,
    int64_t gate_stride,
    int64_t index_stride) {
  using vec_t = aligned_vector_loop<bf16, kVectorWidth>;
  auto* out_vec = reinterpret_cast<vec_t*>(out);
  const auto* gate_vec = reinterpret_cast<const vec_t*>(gate);
  const auto* other_vec = reinterpret_cast<const vec_t*>(other);
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<H3IndexedGateVectorKernel<Index>>(
        sycl::nd_range<1>(sycl::range<1>(rows * kVectorThreads), sycl::range<1>(kVectorThreads)),
        [=](sycl::nd_item<1> item) {
          const int64_t row = item.get_group(0);
          const int64_t col0 = item.get_local_id(0);
          const int64_t gate_row = static_cast<int64_t>(indices[row * index_stride]);
          const int64_t x_base = row * hidden / kVectorWidth;
          const int64_t gate_base = gate_row * gate_stride / kVectorWidth;
          const int64_t bound = hidden / kVectorWidth;
          for (int64_t col = col0; col < bound; col += kVectorThreads) {
            vec_t result = out_vec[x_base + col];
            const vec_t gate_value = gate_vec[gate_base + col];
            const vec_t other_value = other_vec[x_base + col];
#pragma unroll
            for (int i = 0; i < kVectorWidth; ++i) {
              const bf16 product = static_cast<bf16>(
                  static_cast<float>(gate_value[i]) * static_cast<float>(other_value[i]));
              result[i] = static_cast<bf16>(static_cast<float>(result[i]) + static_cast<float>(product));
            }
            out_vec[x_base + col] = result;
          }
        });
  });
}
}  // namespace

SGL_KERNEL_EXPORT void h3_indexed_gate(
    at::Tensor& out, const at::Tensor& x, const at::Tensor& gate, const at::Tensor& other, const at::Tensor& indices) {
  TORCH_CHECK(
      x.is_xpu() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous() && x.size(0) > 0 &&
          x.size(1) > 0 && x.size(1) <= 8192,
      "H3 indexed gate requires contiguous BF16 XPU rows");
  TORCH_CHECK(
      out.device() == x.device() && out.scalar_type() == x.scalar_type() && out.sizes() == x.sizes() &&
          out.is_contiguous(),
      "H3 indexed gate output mismatch");
  TORCH_CHECK(
      other.device() == x.device() && other.scalar_type() == x.scalar_type() && other.sizes() == x.sizes() &&
          other.is_contiguous(),
      "H3 indexed gate other mismatch");
  TORCH_CHECK(
      gate.device() == x.device() && gate.scalar_type() == x.scalar_type() && gate.dim() == 2 &&
          gate.size(1) == x.size(1) && gate.stride(1) == 1,
      "H3 indexed gate table mismatch");
  TORCH_CHECK(
      indices.device() == x.device() && indices.dim() == 1 && indices.numel() == x.size(0) &&
          (indices.scalar_type() == at::kLong || indices.scalar_type() == at::kInt),
      "H3 indexed gate indices mismatch");
  c10::DeviceGuard guard(x.device());
  auto& queue = at::xpu::getCurrentXPUStream().queue();
  const auto* x_ptr = reinterpret_cast<const bf16*>(x.data_ptr<at::BFloat16>());
  const auto* gate_ptr = reinterpret_cast<const bf16*>(gate.data_ptr<at::BFloat16>());
  const auto* other_ptr = reinterpret_cast<const bf16*>(other.data_ptr<at::BFloat16>());
  auto* out_ptr = reinterpret_cast<bf16*>(out.data_ptr<at::BFloat16>());
  constexpr uintptr_t kVectorAlignment = 16;
  const bool vectorizable = out_ptr == x_ptr && x.size(1) % kVectorWidth == 0 &&
      gate.stride(0) % kVectorWidth == 0 &&
      reinterpret_cast<uintptr_t>(other_ptr) % kVectorAlignment == 0 &&
      reinterpret_cast<uintptr_t>(gate_ptr) % kVectorAlignment == 0 &&
      reinterpret_cast<uintptr_t>(out_ptr) % kVectorAlignment == 0;
  if (indices.scalar_type() == at::kLong) {
    if (vectorizable) {
      launch_h3_indexed_gate_vector(queue, gate_ptr, other_ptr, indices.data_ptr<int64_t>(), out_ptr, x.size(0),
                                    x.size(1), gate.stride(0), indices.stride(0));
    } else {
      launch_h3_indexed_gate(queue, x_ptr, gate_ptr, other_ptr, indices.data_ptr<int64_t>(), out_ptr, x.size(0),
                             x.size(1), gate.stride(0), indices.stride(0));
    }
  } else {
    if (vectorizable) {
      launch_h3_indexed_gate_vector(queue, gate_ptr, other_ptr, indices.data_ptr<int32_t>(), out_ptr, x.size(0),
                                    x.size(1), gate.stride(0), indices.stride(0));
    } else {
      launch_h3_indexed_gate(queue, x_ptr, gate_ptr, other_ptr, indices.data_ptr<int32_t>(), out_ptr, x.size(0),
                             x.size(1), gate.stride(0), indices.stride(0));
    }
  }
}
}  // namespace at::native::xpu
