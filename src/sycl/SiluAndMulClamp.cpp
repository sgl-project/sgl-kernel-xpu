// SYCL kernel for fused SiLU-and-Mul with BF16 clamping (DeepSeek-V4 / DeepGEMM style).
//
// Input layout:  [M, 2H]  -- gate is input[:, :H], up is input[:, H:]  (split, not interleaved)
// Output layout: [M, H]   -- written in-place into the caller-provided buffer
//
// Algorithm per output element (m, j):
//   gate_f = (float) input[m, j]
//   up_f   = (float) input[m, H + j]
//   Cast both to bf16, apply clamps (matching CUDA __hmin2 / __hmax2 semantics):
//     gate_c = (float)(bf16) min(gate_f, limit)            -- upper clamp only
//     up_c   = (float)(bf16) clamp(up_f, -limit, limit)   -- both sides
//   SiLU(gate_c) * up_c in fp32:
//     sig    = 1 / (1 + exp(-gate_c))
//     result = gate_c * sig * up_c
//   output[m, j] = (scalar_t) result
//
// Reference: silu_and_mul_clamp_torch() in tests/test_silu_and_mul_clamp.py
//            DeepGEMM mega-MoE (sm100_fp8_fp4_mega_moe.cuh#L984-L997)
//
// Performance design mirrors TripleOps.cpp:
//   - One work-group per token row  →  no per-element integer division
//   - Device-adaptive work-group size via dpcppMaxWorkGroupSize()
//   - Vectorised loads/stores via aligned_vector_loop<scalar_t, N> (N up to 16)

#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "MemoryAccess.h"
#include "SYCLHelpers.h"
#include "Utils.h"

static constexpr float LOG2E = 1.442695040888963f;  // log2(e) for exp2 conversion

//----------------- set element type options --------------------//

template <typename T>
struct ToSyclElementType {
  using type = T;
};

template <>
struct ToSyclElementType<at::Half> {
  using type = sycl::half;
};

template <>
struct ToSyclElementType<at::BFloat16> {
  using type = sycl::ext::oneapi::bfloat16;
};

// ---------------------------------------------------------------------------
// Per-element compute: clamp both halves in bf16, then SiLU(gate)*up in fp32.
// Always clamps through bf16 regardless of input dtype to match the DeepGEMM
// reference (__hmin2/__hmax2 semantics).
// ---------------------------------------------------------------------------
template <typename scalar_t>
inline scalar_t silu_and_mul_clamp_elem(scalar_t a, scalar_t b, float limit) {
  using bf16_t = sycl::ext::oneapi::bfloat16;
  const float limit_bf16 = static_cast<float>(static_cast<bf16_t>(limit));

  // Round-trip through bf16 for clamp (critical: matches DeepGEMM reference)
  const float gate_c = static_cast<float>(
      static_cast<bf16_t>(sycl::fmin(static_cast<float>(static_cast<bf16_t>(static_cast<float>(a))), limit_bf16)));
  const float up_c = static_cast<float>(static_cast<bf16_t>(
      sycl::fmax(-limit_bf16, sycl::fmin(static_cast<float>(static_cast<bf16_t>(static_cast<float>(b))), limit_bf16))));

  // SiLU(gate) * up  in fp32
  const float sig = sycl::native::recip(1.0f + sycl::native::exp2(-gate_c * LOG2E));
  return static_cast<scalar_t>(gate_c * sig * up_c);
}

// ---------------------------------------------------------------------------
// Vectorised kernel functor (follows TripleOps.cpp / op_and_mul_functor).
//
// Grid:  num_group = M (one work-group per token row)
//        wg_size   = min(H/N, max_wg_size)
// Each work-item processes N output elements per iteration, stepping by wg_size.
// ---------------------------------------------------------------------------
template <typename scalar_t, int N>
struct SiluAndMulClampFunctor {
  scalar_t* input_ptr;   // [M, 2H]  (split: gate first, up second)
  scalar_t* output_ptr;  // [M, H]
  int64_t dim;           // H (output hidden dim, = input.size(-1)/2)
  float limit;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t offset = item.get_local_linear_id();
    const int64_t step = item.get_local_range(0);
    const int64_t token_id = item.get_group(0);
    const int64_t bound = dim / N;  // number of vector units in one half-row

    for (int64_t i = offset; i < bound; i += step) {
      // Gate half: input[token_id, :H]  vectorised at position i
      auto gate_vec = reinterpret_cast<const aligned_vector_loop<scalar_t, N>*>(input_ptr)[token_id * bound * 2 + i];
      // Up half:  input[token_id, H:]  vectorised at position i
      auto up_vec =
          reinterpret_cast<const aligned_vector_loop<scalar_t, N>*>(input_ptr)[token_id * bound * 2 + i + bound];

      aligned_vector_loop<scalar_t, N> out_vec;
#pragma unroll
      for (int k = 0; k < N; ++k) {
        out_vec[k] = silu_and_mul_clamp_elem<scalar_t>(gate_vec[k], up_vec[k], limit);
      }

      reinterpret_cast<aligned_vector_loop<scalar_t, N>*>(output_ptr)[token_id * bound + i] = out_vec;
    }
  }
};

// ---------------------------------------------------------------------------
// Launch helper: picks vec_size and wg_size matching TripleOps get_config().
// ---------------------------------------------------------------------------
template <typename TensorDType>
static void silu_and_mul_clamp_sycl(sycl::queue& q, at::Tensor& input, at::Tensor& output, float limit) {
  using KernelDType = typename ToSyclElementType<TensorDType>::type;
  auto* _input = reinterpret_cast<KernelDType*>(input.data_ptr<TensorDType>());
  auto* _output = reinterpret_cast<KernelDType*>(output.data_ptr<TensorDType>());

  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg = dpcppMaxWorkGroupSize(dev_id);
  int64_t numel = output.numel();  // M * H
  int64_t dim = output.size(-1);   // H
  int64_t tokens = numel / dim;    // M
  int64_t num_group = tokens;

  // Intel XPU requires the work-group size to be a multiple of the sub-group
  // size (8, 16, or 32).  Use the largest power-of-2 <= min(H, max_wg) so
  // that H(any non-power-of-2 dim) does not cause a SYCL runtime
  // error from an incompatible work-group size.
  int64_t wg_size = 1;
  {
    const int64_t wg_cap = std::min(dim, max_wg);
    while ((wg_size << 1) <= wg_cap)
      wg_size <<= 1;
  }

  // Compute vector width: start at sizeof(float)*4/sizeof(T), reduce until
  // (vec_size/2)*wg_size < dim  (same heuristic as TripleOps get_config).
  int vec_size = sizeof(float) * 4 / sizeof(KernelDType);
  while ((vec_size >> 1) * wg_size >= dim) {
    vec_size >>= 1;
  }
  if (dim % vec_size != 0) vec_size = 1;

#define VEC_LAUNCH_CLAMP(N)                                                      \
  case N: {                                                                      \
    SiluAndMulClampFunctor<KernelDType, N> kfn = {                               \
        .input_ptr = _input, .output_ptr = _output, .dim = dim, .limit = limit}; \
    sycl_kernel_submit(num_group* wg_size, wg_size, q, kfn);                     \
    break;                                                                       \
  }

  switch (vec_size) {
    VEC_LAUNCH_CLAMP(1);
    VEC_LAUNCH_CLAMP(2);
    VEC_LAUNCH_CLAMP(4);
    VEC_LAUNCH_CLAMP(8);
    VEC_LAUNCH_CLAMP(16);
    default:
      TORCH_CHECK(false, "silu_and_mul_clamp: unsupported vec_size ", vec_size);
  }
#undef VEC_LAUNCH_CLAMP
}

void silu_and_mul_clamp(torch::Tensor& output, torch::Tensor& input, double swiglu_limit) {
  CHECK_INPUT(input)
  CHECK_INPUT(output)
  TORCH_CHECK(output.dtype() == input.dtype(), "silu_and_mul_clamp: dtype mismatch");
  TORCH_CHECK(input.size(-1) % 2 == 0, "silu_and_mul_clamp: input last dim must be even");
  TORCH_CHECK(output.numel() * 2 == input.numel(), "silu_and_mul_clamp: output numel must be half of input numel");
  TORCH_CHECK(swiglu_limit > 0.0, "silu_and_mul_clamp: swiglu_limit must be > 0");

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  float limit = static_cast<float>(swiglu_limit);

  if (input.scalar_type() == at::ScalarType::Half) {
    silu_and_mul_clamp_sycl<at::Half>(queue, input, output, limit);
  } else if (input.scalar_type() == at::ScalarType::BFloat16) {
    silu_and_mul_clamp_sycl<at::BFloat16>(queue, input, output, limit);
  } else {
    TORCH_CHECK(
        input.dtype() == torch::kBFloat16 || input.dtype() == torch::kFloat16,
        "silu_and_mul_clamp: only bf16 and fp16 are supported, got ",
        input.dtype());
  }
}
