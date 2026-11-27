#undef SYCL_INTEL_TARGET
#define SYCL_INTEL_TARGET 20

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>

#include "kernels/gemm/xe20/int8_scaled_mm/int8_scaled_mm.hpp"
#include "sgl_kernel_export.h"

namespace {

inline bool aligned64(const at::Tensor& t) {
  return reinterpret_cast<uintptr_t>(t.data_ptr()) % 64 == 0;
}

}  // namespace

SGL_KERNEL_EXPORT torch::Tensor int8_scaled_mm(
    const torch::Tensor& mat_a,
    const torch::Tensor& mat_b,
    const torch::Tensor& scales_a,
    const torch::Tensor& scales_b,
    const torch::Dtype& out_dtype,
    const c10::optional<torch::Tensor>& bias) {
  TORCH_CHECK(mat_a.is_xpu(), "int8_scaled_mm: mat_a must be an XPU tensor");
  TORCH_CHECK(mat_b.is_xpu(), "int8_scaled_mm: mat_b must be an XPU tensor");
  TORCH_CHECK(
      mat_a.device() == mat_b.device() && mat_a.device() == scales_a.device() && mat_a.device() == scales_b.device(),
      "int8_scaled_mm: all tensors must be on the same XPU device");
  TORCH_CHECK(mat_a.dim() == 2, "int8_scaled_mm: mat_a must be a 2D tensor");
  TORCH_CHECK(mat_b.dim() == 2, "int8_scaled_mm: mat_b must be a 2D tensor");
  TORCH_CHECK(mat_a.stride(1) == 1, "int8_scaled_mm: mat_a must be row-major");
  TORCH_CHECK(mat_b.stride(0) == 1, "int8_scaled_mm: mat_b must be column-major");
  TORCH_CHECK(mat_a.scalar_type() == torch::kInt8, "int8_scaled_mm: mat_a must be Int8");
  TORCH_CHECK(mat_b.scalar_type() == torch::kInt8, "int8_scaled_mm: mat_b must be Int8");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(0), "int8_scaled_mm: mat_a and mat_b inner dimensions must match");
  TORCH_CHECK(
      out_dtype == torch::kHalf || out_dtype == torch::kBFloat16, "int8_scaled_mm: out_dtype must be Half or BFloat16");

  const int64_t M = mat_a.size(0);
  const int64_t K = mat_a.size(1);
  const int64_t N = mat_b.size(1);

  TORCH_CHECK(K % 4 == 0, "int8_scaled_mm: K must be a multiple of 4 for block-2D access, got ", K);
  TORCH_CHECK(N % 2 == 0, "int8_scaled_mm: N must be a multiple of 2 for block-2D access, got ", N);
  TORCH_CHECK(
      M < (1 << 24) && N < (1 << 23) && K < (1 << 24),
      "int8_scaled_mm: matrix dimensions exceed Xe2 block-2D surface limits");

  TORCH_CHECK(scales_a.numel() == M, "int8_scaled_mm: size of scales_a does not match M");
  TORCH_CHECK(scales_b.numel() == N, "int8_scaled_mm: size of scales_b does not match N");
  TORCH_CHECK(scales_a.scalar_type() == torch::kFloat32, "int8_scaled_mm: scales_a must be Float32");
  TORCH_CHECK(scales_b.scalar_type() == torch::kFloat32, "int8_scaled_mm: scales_b must be Float32");

  if (bias.has_value()) {
    TORCH_CHECK(bias->device() == mat_a.device(), "int8_scaled_mm: bias must be on the same device");
    TORCH_CHECK(bias->numel() == N, "int8_scaled_mm: size of bias does not match N");
    TORCH_CHECK(bias->scalar_type() == out_dtype, "int8_scaled_mm: bias dtype must match out_dtype");
  }

  torch::Tensor out = torch::empty({M, N}, mat_a.options().dtype(out_dtype));
  if (M == 0 || N == 0) {
    return out;
  }
  if (K == 0) {
    if (bias.has_value()) {
      out.copy_(bias->reshape({1, N}).expand({M, N}));
    } else {
      out.zero_();
    }
    return out;
  }

  c10::DeviceGuard guard(mat_a.device());

  at::Tensor a = mat_a;
  if (!aligned64(a) || (a.stride(0) % 4 != 0)) {
    a = a.contiguous();
    if (!aligned64(a)) a = a.clone();
  }
  at::Tensor b_t = mat_b.t();  // [N, K] row-major view of column-major mat_b [K, N]
  if (!aligned64(b_t) || (b_t.stride(0) % 4 != 0)) {
    b_t = b_t.contiguous();
    if (!aligned64(b_t)) b_t = b_t.clone();
  }
  at::Tensor sa = scales_a.contiguous();
  at::Tensor sb = scales_b.contiguous();
  at::Tensor bias_contig;
  if (bias.has_value()) {
    bias_contig = bias->contiguous();
  }

  sycl::queue& queue = c10::xpu::getCurrentXPUStream(mat_a.device().index()).queue();

  int8_scaled_mm_xe20::run(
      queue,
      out.data_ptr(),
      a.data_ptr<int8_t>(),
      b_t.data_ptr<int8_t>(),
      sa.data_ptr<float>(),
      sb.data_ptr<float>(),
      bias.has_value() ? bias_contig.data_ptr() : nullptr,
      static_cast<int>(M),
      static_cast<int>(N),
      static_cast<int>(K),
      static_cast<int>(a.stride(0)),
      static_cast<int>(b_t.stride(0)),
      static_cast<int>(out.stride(0)),
      out_dtype == torch::kBFloat16);

  return out;
}

#undef SYCL_INTEL_TARGET
