/**
 * Activation-and-Mul SYCL JIT Kernels for SGLang XPU
 *
 * Fused gated activations used by SwiGLU/GeGLU MLPs. Input is split (not
 * interleaved): gate = input[:, :dim], up = input[:, dim:].
 *
 *   silu_and_mul(x)      : out = silu(gate) * up
 *   gelu_and_mul(x)      : out = gelu_erf(gate) * up
 *   gelu_tanh_and_mul(x) : out = gelu_tanh(gate) * up
 *
 *   Input  : [num_tokens, 2*dim]   (native dtype)
 *   Output : [num_tokens, dim]     (native dtype)
 *
 * Self-contained JIT port of the AOT kernels in src/sycl/TripleOps.cpp. Kept
 * numerically identical (same functor math, fp32 accumulation):
 *   - one work-group per token row, grid-stride over vector units in the row
 *   - vectorized load/store via aligned_vector<T, N>
 *   - vec width + work-group size chosen with the AOT get_config() heuristic
 *   - no .wait() in the launcher (PyTorch owns stream synchronization)
 *
 * A single exported symbol per dtype takes a runtime `act_kind`, so one compiled
 * .so per dtype serves all three activations (mirrors the CUDA JIT
 * run_activation(op_name, ...) design in jit_kernel/csrc/elementwise/activation.cuh).
 */

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

#include "../memory.hpp"

namespace sgl {
namespace sycl_kernel {

using sgl::sycl::aligned_vector;

// Selects which activation is applied to the gate half. Values must match
// _ACT_KIND_MAP in the Python wrapper.
enum class ActivationKind : int32_t {
  kSiLU = 0,
  kGELU = 1,      // erf-based (exact) GELU
  kGELUTanh = 2,  // tanh approximation
};

// Per-element gated activation: act(gate) * up, computed in fp32 (matches the
// at::opmath_type<scalar_t> accumulation in the AOT functors).
template <typename T, ActivationKind kAct>
inline T act_and_mul_elem(T gate, T up) {
  const float a = static_cast<float>(gate);
  const float b = static_cast<float>(up);
  float y;
  if constexpr (kAct == ActivationKind::kSiLU) {
    y = a / (1.0f + ::sycl::exp(-a));
  } else if constexpr (kAct == ActivationKind::kGELU) {
    constexpr float kSqrt1Over2 = 0.7071067811865476f;  // 1/sqrt(2) (M_SQRT1_2)
    y = a * 0.5f * (1.0f + ::sycl::erf(a * kSqrt1Over2));
  } else {                                        // kGELUTanh
    constexpr float kBeta = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float kKappa = 0.044715f;
    const float x_cube = a * a * a;
    const float inner = kBeta * (a + kKappa * x_cube);
    y = 0.5f * a * (1.0f + ::sycl::tanh(inner));
  }
  return static_cast<T>(y * b);
}

// ---------------------------------------------------------------------------
// Vectorized kernel functor. Grid: num_group = num_tokens (one WG per row),
// each work-item handles N output elements per iteration, stepping by wg_size.
// ---------------------------------------------------------------------------
template <typename T, ActivationKind kAct, int N>
class ActAndMulKernel {
 public:
  using Vec = aligned_vector<T, N>;

  ActAndMulKernel(const T* input, T* output, int64_t dim) : input_(input), output_(output), dim_(dim) {}

  void operator()(::sycl::nd_item<1> item) const {
    const int64_t offset = item.get_local_linear_id();
    const int64_t step = item.get_local_range(0);
    const int64_t token_id = item.get_group(0);
    const int64_t bound = dim_ / N;  // vector units in one half-row (gate/up)

    // Row layout in vector units: [gate(bound) | up(bound)].
    const Vec* in_vec = reinterpret_cast<const Vec*>(input_);
    Vec* out_vec = reinterpret_cast<Vec*>(output_);
    const int64_t in_row_base = token_id * bound * 2;
    const int64_t out_row_base = token_id * bound;

    for (int64_t i = offset; i < bound; i += step) {
      const Vec gate_v = in_vec[in_row_base + i];
      const Vec up_v = in_vec[in_row_base + i + bound];

      Vec out_v;
#pragma unroll
      for (int k = 0; k < N; ++k) {
        out_v[k] = act_and_mul_elem<T, kAct>(gate_v[k], up_v[k]);
      }
      out_vec[out_row_base + i] = out_v;
    }
  }

 private:
  const T* input_;
  T* output_;
  int64_t dim_;
};

// ---------------------------------------------------------------------------
// Host launcher. Picks vec width + work-group size with the AOT get_config()
// heuristic, then dispatches on (act_kind, vec_size).
// ---------------------------------------------------------------------------
template <typename T>
void act_and_mul_launcher(
    ::sycl::queue& queue, const void* input, void* output, int64_t num_tokens, int64_t dim, int32_t act_kind) {
  if (num_tokens <= 0 || dim <= 0) return;

  const T* input_ptr = static_cast<const T*>(input);
  T* output_ptr = static_cast<T*>(output);

  // Largest power-of-2 work-group size <= min(dim, 1024). A power of two keeps
  // it a multiple of the sub-group size for any (non-pow2) dim.
  constexpr int64_t kMaxWg = 1024;
  int64_t wg_size = 1;
  {
    const int64_t wg_cap = ::sycl::min(dim, kMaxWg);
    while ((wg_size << 1) <= wg_cap)
      wg_size <<= 1;
  }

  // vec_size = 16 bytes worth of elements, shrunk until each work-item has work
  // (matches AOT get_config), then scalar fallback if dim is not divisible.
  int vec_size = static_cast<int>(sizeof(float) * 4 / sizeof(T));
  while (vec_size > 1 && (vec_size >> 1) * wg_size >= dim) {
    vec_size >>= 1;
  }
  if (dim % vec_size != 0) vec_size = 1;

  const int64_t num_group = num_tokens;
  const ::sycl::nd_range<1> range(::sycl::range<1>(num_group * wg_size), ::sycl::range<1>(wg_size));

#define SGL_ACT_LAUNCH(KIND, NVAL)                                                       \
  queue.submit([&](::sycl::handler& cgh) {                                               \
    cgh.parallel_for(range, ActAndMulKernel<T, KIND, NVAL>(input_ptr, output_ptr, dim)); \
  });

#define SGL_ACT_DISPATCH_VEC(KIND)    \
  switch (vec_size) {                 \
    case 1:                           \
      SGL_ACT_LAUNCH(KIND, 1) break;  \
    case 2:                           \
      SGL_ACT_LAUNCH(KIND, 2) break;  \
    case 4:                           \
      SGL_ACT_LAUNCH(KIND, 4) break;  \
    case 8:                           \
      SGL_ACT_LAUNCH(KIND, 8) break;  \
    default:                          \
      SGL_ACT_LAUNCH(KIND, 16) break; \
  }

  switch (static_cast<ActivationKind>(act_kind)) {
    case ActivationKind::kSiLU:
      SGL_ACT_DISPATCH_VEC(ActivationKind::kSiLU)
      break;
    case ActivationKind::kGELU:
      SGL_ACT_DISPATCH_VEC(ActivationKind::kGELU)
      break;
    case ActivationKind::kGELUTanh:
      SGL_ACT_DISPATCH_VEC(ActivationKind::kGELUTanh)
      break;
  }
#undef SGL_ACT_DISPATCH_VEC
#undef SGL_ACT_LAUNCH
  // NOTE: no .wait() -- PyTorch handles stream synchronization.
}

// ---------------------------------------------------------------------------
// C API for the Python ctypes wrapper. One symbol per dtype; the activation is
// selected at runtime via act_kind.
// ---------------------------------------------------------------------------
#define DEFINE_ACT_AND_MUL_FORWARD(DTYPE_SUFFIX, DTYPE)                                                      \
  extern "C" void act_and_mul_forward_##DTYPE_SUFFIX(                                                        \
      void* queue_ptr, const void* input, void* output, int64_t num_tokens, int64_t dim, int32_t act_kind) { \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr);                                                   \
    act_and_mul_launcher<DTYPE>(queue, input, output, num_tokens, dim, act_kind);                            \
  }

#if defined(SGL_ACT_DTYPE_fp32)
DEFINE_ACT_AND_MUL_FORWARD(fp32, float)
#elif defined(SGL_ACT_DTYPE_fp16)
DEFINE_ACT_AND_MUL_FORWARD(fp16, ::sycl::half)
#elif defined(SGL_ACT_DTYPE_bf16)
DEFINE_ACT_AND_MUL_FORWARD(bf16, ::sycl::ext::oneapi::bfloat16)
#else
DEFINE_ACT_AND_MUL_FORWARD(fp32, float)
DEFINE_ACT_AND_MUL_FORWARD(fp16, ::sycl::half)
DEFINE_ACT_AND_MUL_FORWARD(bf16, ::sycl::ext::oneapi::bfloat16)
#endif

#undef DEFINE_ACT_AND_MUL_FORWARD

}  // namespace sycl_kernel
}  // namespace sgl
