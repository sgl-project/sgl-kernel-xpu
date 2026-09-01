#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cstdint>
#include <limits>
#include <sycl/sycl.hpp>

#include "MemoryAccess.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

// gate * sigmoid(gate * alpha) * (up + 1), evaluated in fp32.
//
// This must stay expression-for-expression identical to the fused BF16
// grouped-GEMM epilogue (kernels/moe/xe20/common/activation.hpp:36-40), which
// is the other implementation of the same activation: same clamp order (upper
// clamp only on gate, symmetric clamp on up), same native::exp sigmoid, same
// (up + 1) factor. For 2-byte outputs native::exp's ~1e-6 relative error sits
// ~3 decades below the 2^-8 bf16 / 2^-11 fp16 output quantum, so it is not
// observable after the narrowing cast. fp32 outputs keep the precise exp,
// where 1e-6 would be ~10 ULP of the result -- and the fused epilogue is
// bf16-only, so there is no fp32 parity requirement to trade against.
template <typename scalar_t>
inline float swiglu_gpt_oss_elem(float gate, float up, float alpha, float limit) {
  gate = sycl::fmin(gate, limit);
  up = sycl::fmax(-limit, sycl::fmin(up, limit));

  const float t = gate * alpha;
  float sig;
  if constexpr (sizeof(scalar_t) == 2) {
    sig = 1.0f / (1.0f + sycl::native::exp(-t));
  } else {
    sig = 1.0f / (1.0f + sycl::exp(-t));
  }
  return gate * sig * (up + 1.0f);
}

}  // namespace

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
    y[idx] = static_cast<scalar_t>(swiglu_gpt_oss_elem<scalar_t>(
        static_cast<float>(x[2 * idx]), static_cast<float>(x[2 * idx + 1]), gemm1_alpha, gemm1_limit));
  }
};

// Interleaved-pair vector kernel.
//
// x is [B, 2H] with gate at 2*i and up at 2*i+1; y is [B, H]. One work-item
// owns one contiguous VP-wide slice of y, which is exactly one contiguous
// 2*VP-wide slice of x: a single 32 B block load and a single 16 B block
// store, with the gate/up de-interleave done in registers.
//
// This is the point of the rewrite. The previous kernel indexed x per element
// as x[2 * (base + i)], which makes the address an affine function of the lane
// with a stride of 2 * sizeof(scalar_t) -- IGC has to emit that as a
// 16-address gather, and the VP scalar stores likewise as scatters. Loading
// the interleaved pair-block whole keeps every access a full-width contiguous
// transfer across the sub-group.
//
// There is no contiguous gate-only slice to load instead: gate lives only at
// even indices, so a "gate vector load" is a stride-2 gather by construction.
// The pair-block is the only formulation where both load and store are
// full-width contiguous.
//
// aligned_vector_loop (MemoryAccess.h:73) rather than sycl::vec<scalar_t, 16>
// because it is the in-tree idiom for exactly this (SiluAndMulClamp.cpp:99)
// and is a bare alignas POD array with no element-type traits to satisfy.
template <typename scalar_t, int VP>
struct SwigluInterleavedVecKernel {
  using in_vec_t = aligned_vector_loop<scalar_t, 2 * VP>;
  using out_vec_t = aligned_vector_loop<scalar_t, VP>;

  const scalar_t* x;
  scalar_t* y;
  std::int64_t vec_count;    // whole VP-wide output vectors
  std::int64_t total_pairs;  // B * H
  float gemm1_alpha;
  float gemm1_limit;

  SwigluInterleavedVecKernel(
      const scalar_t* x_, scalar_t* y_, std::int64_t vec_count_, std::int64_t total_pairs_, float alpha_, float limit_)
      : x(x_), y(y_), vec_count(vec_count_), total_pairs(total_pairs_), gemm1_alpha(alpha_), gemm1_limit(limit_) {}

  inline void operator()(sycl::nd_item<1> it) const {
    const std::int64_t vec_idx = it.get_global_linear_id();

    if (vec_idx < vec_count) {
      const in_vec_t in_v = reinterpret_cast<const in_vec_t*>(x)[vec_idx];

      out_vec_t out_v;
#pragma unroll
      for (int k = 0; k < VP; ++k) {
        out_v[k] = static_cast<scalar_t>(swiglu_gpt_oss_elem<scalar_t>(
            static_cast<float>(in_v[2 * k]), static_cast<float>(in_v[2 * k + 1]), gemm1_alpha, gemm1_limit));
      }

      reinterpret_cast<out_vec_t*>(y)[vec_idx] = out_v;
      return;
    }

    // Tail. The one work-item immediately past the vector range finishes the
    // fewer-than-VP leftover pairs elementwise, so the op stays at a single
    // launch and the bound is checked per element rather than per block. The
    // previous kernel checked `vec_idx * 4 >= total_pairs`, which is
    // block-granular: a work-item with base < total_pairs <= base + 3 passed
    // the guard and then read/wrote up to 3 pairs past the end of x and y.
    if (vec_idx == vec_count) {
      for (std::int64_t i = vec_count * VP; i < total_pairs; ++i) {
        y[i] = static_cast<scalar_t>(swiglu_gpt_oss_elem<scalar_t>(
            static_cast<float>(x[2 * i]), static_cast<float>(x[2 * i + 1]), gemm1_alpha, gemm1_limit));
      }
    }
  }
};

template <typename scalar_t>
void swiglu_gpt_oss_sigmoid_alpha_sycl(
    const scalar_t* x,  // [B, 2*H]
    scalar_t* y,        // [B, H]
    size_t batch,
    size_t hidden,
    float alpha,
    float limit) {
  // 16 B of output per work-item -- 8 elements at bf16/fp16, 4 at fp32 -- and
  // therefore a 32 B input block either way. Both are single-instruction
  // widths on Xe.
  constexpr int VP = 16 / sizeof(scalar_t);

  const std::int64_t total_pairs = static_cast<std::int64_t>(batch) * static_cast<std::int64_t>(hidden);
  if (total_pairs == 0) return;

  const size_t local = 256;
  auto stream = at::xpu::getCurrentXPUStream();
  auto q = stream.queue();

  // The vector path reinterpret_casts both pointers, so it needs the vector
  // alignment. Every in-tree caller passes freshly allocated (>= 256 B aligned)
  // tensors; a contiguous view with an unaligned storage_offset falls back to
  // the scalar kernel rather than being rejected.
  const bool can_vectorize = (reinterpret_cast<std::uintptr_t>(x) % (2 * VP * sizeof(scalar_t)) == 0) &&
                             (reinterpret_cast<std::uintptr_t>(y) % (VP * sizeof(scalar_t)) == 0);

  if (!can_vectorize) {
    const size_t global = ((static_cast<size_t>(total_pairs) + local - 1) / local) * local;
    q.submit([&](sycl::handler& h) {
      SwigluScalarKernel<scalar_t> kernel_functor(x, y, total_pairs, alpha, limit);
      h.parallel_for(sycl::nd_range<1>(global, local), kernel_functor);
    });
    return;
  }

  const std::int64_t vec_count = total_pairs / VP;
  const std::int64_t work_items = vec_count + ((total_pairs % VP) != 0 ? 1 : 0);
  const size_t global = ((static_cast<size_t>(work_items) + local - 1) / local) * local;

  q.submit([&](sycl::handler& h) {
    SwigluInterleavedVecKernel<scalar_t, VP> kernel_functor(x, y, vec_count, total_pairs, alpha, limit);
    h.parallel_for(sycl::nd_range<1>(global, local), kernel_functor);
  });
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

#define CALL_SWIGLU_LAUNCHER_SYCL(scalar_t)                                                                \
  {                                                                                                        \
    swiglu_gpt_oss_sigmoid_alpha_sycl<scalar_t>(                                                           \
        reinterpret_cast<const scalar_t*>(x_ptr), reinterpret_cast<scalar_t*>(y_ptr), B, H, alpha, limit); \
  }

SGL_KERNEL_EXPORT torch::Tensor swiglu_gpt_oss_sigmoid_alpha(
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

  SYCL_DISPATCH_BY_SCALAR_DTYPE(x.scalar_type(), CALL_SWIGLU_LAUNCHER_SYCL);

  return y;
}
