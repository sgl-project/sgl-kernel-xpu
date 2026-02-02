#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <limits>
#include <sycl/sycl.hpp>

#include "Utils.h"

template <typename scalar_t>
struct SwigluScalarKernel {
  const scalar_t* x;
  scalar_t* y;
  std::int64_t total_pairs;
  float gemm1_alpha;
  float gemm1_limit;

  SwigluScalarKernel(const scalar_t* x_, scalar_t* y_, std::int64_t total_pairs_, float alpha_, float limit_)
      : x(x_), y(y_), total_pairs(total_pairs_), gemm1_alpha(alpha_), gemm1_limit(limit_) {}

  inline void operator()(sycl::nd_item<1> it) const {
    const std::int64_t idx = it.get_global_linear_id();
    if (idx >= total_pairs) return;

    // x layout: [..., 2*i] = gate, [..., 2*i+1] = up
    const scalar_t gate_raw = x[2 * idx];
    const scalar_t up_raw = x[2 * idx + 1];

    // work in float for math stability/precision
    float gate = static_cast<float>(gate_raw);
    float up = static_cast<float>(up_raw);

    gate = sycl::fmin(gate, gemm1_limit);

    up = sycl::fmax(-gemm1_limit, sycl::fmin(up, gemm1_limit));

    // gate * sigmoid(gate * gemm1_alpha) * (up + 1)
    const float t = gate * gemm1_alpha;
    const float sig = 1.0f / (1.0f + sycl::exp(-t));
    const float out = gate * sig * (up + 1.0f);

    y[idx] = static_cast<scalar_t>(out);
  }
};

template <typename scalar_t>
struct SwigluVec4Kernel {
  const scalar_t* x;  // [B, 2H]
  scalar_t* y;        // [B, H]
  std::int64_t total_pairs;
  float gemm1_alpha;
  float gemm1_limit;

  SwigluVec4Kernel(const scalar_t* x_, scalar_t* y_, std::int64_t total_pairs_, float alpha_, float limit_)
      : x(x_), y(y_), total_pairs(total_pairs_), gemm1_alpha(alpha_), gemm1_limit(limit_) {}

  inline void operator()(sycl::nd_item<1> it) const {
    const std::int64_t vec_idx = it.get_global_linear_id();
    const std::int64_t base = vec_idx * 4;

    if (base >= total_pairs) return;

    // ---- load 4 gate values ----
    sycl::vec<float, 4> gate_v;
    sycl::vec<float, 4> up_v;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      gate_v[i] = static_cast<float>(x[2 * (base + i)]);
      up_v[i] = static_cast<float>(x[2 * (base + i) + 1]);
    }

// ---- clamp ----
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      gate_v[i] = sycl::fmin(gate_v[i], gemm1_limit);
      up_v[i] = sycl::fmax(-gemm1_limit, sycl::fmin(up_v[i], gemm1_limit));
    }

    // ---- swiglu ----
    sycl::vec<float, 4> out_v;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float t = gate_v[i] * gemm1_alpha;
      const float sig = 1.0f / (1.0f + sycl::exp(-t));
      out_v[i] = gate_v[i] * sig * (up_v[i] + 1.0f);
    }

// ---- store 4 outputs ----
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      y[base + i] = static_cast<scalar_t>(out_v[i]);
    }
  }
};

template <typename scalar_t>
void swiglu_with_alpha_and_limit_sycl(
    const scalar_t* x,  // [B, 2*H]
    scalar_t* y,        // [B, H]
    size_t batch,
    size_t hidden,
    float alpha,
    float limit) {
  const size_t pairs = batch * hidden;  // gate/up pairs
  const size_t vec_pairs = pairs / 4;
  const size_t remainder_start = vec_pairs * 4;
  const size_t remainder = pairs - remainder_start;

  const size_t local = 256;
  auto stream = at::xpu::getCurrentXPUStream();
  auto q = stream.queue();
  if (vec_pairs > 0) {
    const size_t global = ((vec_pairs + local - 1) / local) * local;
    q.submit([&](sycl::handler& h) {
      SwigluVec4Kernel<scalar_t> kernel_functor(x, y, pairs, alpha, limit);  // Pass pairs, not vec_pairs
      h.parallel_for(sycl::nd_range<1>(global, local), kernel_functor);
    });
  }
  if (remainder > 0) {
    const size_t global_rem = ((remainder + local - 1) / local) * local;
    q.submit([&](sycl::handler& h) {
      SwigluScalarKernel<scalar_t> kernel_functor(
          x + 2 * remainder_start, y + remainder_start, remainder, alpha, limit);
      h.parallel_for(sycl::nd_range<1>(global_rem, local), kernel_functor);
    });
  }
}

#define SYCL_DISPATCH_BY_SCALAR_DTYPE(scalar_dtype, fn)                    \
  {                                                                        \
    if (scalar_dtype == at::ScalarType::Float) {                           \
      fn(float);                                                           \
    } else if (scalar_dtype == at::ScalarType::Half) {                     \
      fn(sycl::half);                                                      \
    } else if (scalar_dtype == at::ScalarType::BFloat16) {                 \
      fn(sycl::ext::oneapi::bfloat16);                                     \
    } else {                                                               \
      TORCH_CHECK(false, "Unsupported dtype for SYCL op: ", scalar_dtype); \
    }                                                                      \
  }

#define CALL_SWIGLU_VEC4_LAUNCHER_SYCL(scalar_t)                                                           \
  {                                                                                                        \
    swiglu_with_alpha_and_limit_sycl<scalar_t>(                                                            \
        reinterpret_cast<const scalar_t*>(x_ptr), reinterpret_cast<scalar_t*>(y_ptr), B, H, alpha, limit); \
  }

torch::Tensor swiglu_with_alpha_and_limit(
    torch::Tensor x,  // [B, 2H]
    double alpha,
    double limit) {
  TORCH_CHECK(x.is_xpu(), "Unsupported device");
  TORCH_CHECK(
      x.dtype() == torch::kFloat32 || x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16,
      "Only float32, float16, and bfloat16 are supported");
  TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
  TORCH_CHECK(x.dim() == 2, "x must be 2D [B, 2H]");
  TORCH_CHECK(x.size(1) % 2 == 0, "Last dim must be even");

  const int64_t B = x.size(0);
  const int64_t H2 = x.size(1);
  const int64_t H = H2 / 2;

  // output: [B, H]
  auto y = torch::empty({B, H}, x.options());

  const void* x_ptr = x.data_ptr();
  void* y_ptr = y.data_ptr();

  SYCL_DISPATCH_BY_SCALAR_DTYPE(x.scalar_type(), CALL_SWIGLU_VEC4_LAUNCHER_SYCL);

  return y;
}
