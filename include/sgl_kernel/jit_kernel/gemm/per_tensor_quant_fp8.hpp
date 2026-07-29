/**
 * Per-Tensor FP8 Quantization SYCL JIT Kernel for SGLang XPU
 *
 * Quantizes a whole tensor to fp8 (e4m3fn) with a single global scale:
 *
 *   dynamic (is_static=false):
 *     output_s = absmax(input) / 448          (one scalar, cross-block reduce)
 *   then, for every element:
 *     output_q[i] = clamp(input[i] / (output_s + eps), -448, 448)  -> e4m3fn
 *
 * Self-contained JIT port of the AOT kernel in src/sycl/per_tensor_quant_fp8.cpp.
 * Kept numerically close to it:
 *   - grid-stride vectorized load (VEC_SIZE = 16 / sizeof(T)) + scalar tail loop
 *   - fp32 absmax accumulation, block reduce_over_group(maximum), then a
 *     cross-block atomic-CAS max into the single output_s
 *   - eps = 1e-8 guard in the reciprocal (matches AOT)
 *
 * Differences from AOT (behaviorally identical):
 *   - FP8 e4m3fn conversion uses a self-contained c10-style routine instead of
 *     CUTLASS (bit-exact with torch's `.to(torch.float8_e4m3fn)` in the clamped
 *     [-448, 448] range this kernel produces).
 *   - dtype and is_static are compile-time specialized via -D macros. is_static
 *     is baked in with `if constexpr`, so the static .so contains ONLY the
 *     quantize kernel (no absmax) -- mirrors the CUDA JIT per_tensor_quant_fp8.
 */

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace sgl {
namespace sycl_kernel {
namespace per_tensor_quant_fp8 {

static constexpr int kSubGroupSize = 32;
static constexpr int kBlockSize = 256;
static constexpr float kFp8E4M3Max = 448.0f;
static constexpr float kEps = 1e-8f;

// c10-style round-to-nearest-even float -> e4m3fn (finite; NaN at 0x7f/0xff).
// Inputs are pre-clamped to +-448, so the saturation branch is unreached.
inline uint8_t fp32_to_e4m3fn(float f) {
  const uint32_t fp8_max_bits = static_cast<uint32_t>(1087) << 20;
  const uint32_t denorm_mask = static_cast<uint32_t>(141) << 23;

  uint32_t f_bits = ::sycl::bit_cast<uint32_t>(f);
  uint8_t result = 0u;

  const uint32_t sign = f_bits & static_cast<uint32_t>(0x80000000);
  f_bits ^= sign;

  if (f_bits >= fp8_max_bits) {
    result = 0x7f;  // NaN / out-of-range (unreached after clamp)
  } else if (f_bits < (static_cast<uint32_t>(121) << 23)) {
    float shifted = ::sycl::bit_cast<float>(f_bits) + ::sycl::bit_cast<float>(denorm_mask);
    f_bits = ::sycl::bit_cast<uint32_t>(shifted);
    result = static_cast<uint8_t>(f_bits - denorm_mask);
  } else {
    uint8_t mant_odd = (f_bits >> 20) & 1u;
    f_bits += (static_cast<uint32_t>(7 - 127) << 23) + 0x7FFFFu;
    f_bits += mant_odd;
    result = static_cast<uint8_t>(f_bits >> 20);
  }

  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

// ---------------------------------------------------------------------------
// Absmax kernel: grid-stride vectorized abs-max, block reduce, cross-block
// atomic-CAS max into output_s (= global absmax / 448). Dynamic path only.
// ---------------------------------------------------------------------------
template <typename T, int VEC_SIZE>
class PerTensorAbsMaxKernel {
 public:
  PerTensorAbsMaxKernel(const T* input, float* output_s, int64_t num_elements)
      : input_(input), output_s_(output_s), num_elements_(num_elements) {}

  [[sycl::reqd_sub_group_size(kSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
    const int64_t gid = item.get_global_id(0);
    const int64_t grid_size = item.get_global_range(0);
    const int64_t num_vec_elems = num_elements_ / VEC_SIZE;

    using vec_type = ::sycl::vec<T, VEC_SIZE>;
    float max_value = 0.0f;

    for (int64_t i = gid; i < num_vec_elems; i += grid_size) {
      vec_type input_vec;
      input_vec.load(0, ::sycl::multi_ptr<const T, ::sycl::access::address_space::global_space>(input_ + i * VEC_SIZE));
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        max_value = ::sycl::fmax(max_value, ::sycl::fabs(static_cast<float>(input_vec[j])));
      }
    }

    const int64_t remaining_start = num_vec_elems * VEC_SIZE;
    for (int64_t idx = remaining_start + gid; idx < num_elements_; idx += grid_size) {
      max_value = ::sycl::fmax(max_value, ::sycl::fabs(static_cast<float>(input_[idx])));
    }

    max_value = ::sycl::reduce_over_group(item.get_group(), max_value, ::sycl::maximum<>());

    if (item.get_local_id(0) == 0) {
      const float local_scale = max_value / kFp8E4M3Max;
      ::sycl::atomic_ref<
          float,
          ::sycl::memory_order::acq_rel,
          ::sycl::memory_scope::device,
          ::sycl::access::address_space::global_space>
          atomic_scale(*output_s_);
      float global_scale = atomic_scale.load();
      while (global_scale < local_scale && !atomic_scale.compare_exchange_strong(global_scale, local_scale)) {
      }
    }
  }

 private:
  const T* input_;
  float* output_s_;
  int64_t num_elements_;
};

// ---------------------------------------------------------------------------
// Quantize kernel: grid-stride vectorized scale-clip-cast to e4m3fn (uint8).
// ---------------------------------------------------------------------------
template <typename T, int VEC_SIZE>
class PerTensorQuantFP8Kernel {
 public:
  PerTensorQuantFP8Kernel(const T* input, uint8_t* output, const float* scale, int64_t num_elements)
      : input_(input), output_(output), scale_(scale), num_elements_(num_elements) {}

  [[sycl::reqd_sub_group_size(kSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
    const int64_t gid = item.get_global_id(0);
    const int64_t grid_size = item.get_global_range(0);
    const int64_t num_vec_elems = num_elements_ / VEC_SIZE;

    const float scale_val = 1.0f / (*scale_ + kEps);
    using vec_type_in = ::sycl::vec<T, VEC_SIZE>;
    using vec_type_out = ::sycl::vec<uint8_t, VEC_SIZE>;

    for (int64_t i = gid; i < num_vec_elems; i += grid_size) {
      const int64_t base_idx = i * VEC_SIZE;
      vec_type_in input_vec;
      input_vec.load(0, ::sycl::multi_ptr<const T, ::sycl::access::address_space::global_space>(input_ + base_idx));
      vec_type_out output_vec;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        float val = static_cast<float>(input_vec[j]) * scale_val;
        val = ::sycl::fmax(-kFp8E4M3Max, ::sycl::fmin(val, kFp8E4M3Max));
        output_vec[j] = fp32_to_e4m3fn(val);
      }
      output_vec.store(
          0,
          ::sycl::address_space_cast<::sycl::access::address_space::global_space, ::sycl::access::decorated::yes>(
              output_ + base_idx));
    }

    const int64_t remaining_start = num_vec_elems * VEC_SIZE;
    for (int64_t idx = remaining_start + gid; idx < num_elements_; idx += grid_size) {
      float val = static_cast<float>(input_[idx]) * scale_val;
      val = ::sycl::fmax(-kFp8E4M3Max, ::sycl::fmin(val, kFp8E4M3Max));
      output_[idx] = fp32_to_e4m3fn(val);
    }
  }

 private:
  const T* input_;
  uint8_t* output_;
  const float* scale_;
  int64_t num_elements_;
};

// ---------------------------------------------------------------------------
// Host launcher. kIsStatic strips the absmax launch at compile time.
// ---------------------------------------------------------------------------
template <typename T, bool kIsStatic>
void per_tensor_quant_fp8_launcher(
    ::sycl::queue& queue, const void* input, void* output_q, void* output_s, int64_t num_elements) {
  if (num_elements <= 0) return;

  const T* input_ptr = static_cast<const T*>(input);
  uint8_t* output_q_ptr = static_cast<uint8_t*>(output_q);
  float* output_s_ptr = static_cast<float*>(output_s);

  const int64_t block_cap = kIsStatic ? 4096 : 2048;
  const int64_t num_blocks = std::min((num_elements + kBlockSize - 1) / kBlockSize, block_cap);
  const ::sycl::range<1> global_range(static_cast<size_t>(num_blocks) * kBlockSize);
  const ::sycl::range<1> local_range(kBlockSize);

  // vec width: 16 bytes worth of elements, shrunk until it divides num_elements.
  int vec_size = static_cast<int>(sizeof(float) * 4 / sizeof(T));
  while (vec_size > 1 && (num_elements % vec_size != 0)) {
    vec_size >>= 1;
  }

#define SGL_PTQ_LAUNCH(NVAL)                                                                      \
  do {                                                                                            \
    if constexpr (!kIsStatic) {                                                                   \
      queue.submit([&](::sycl::handler& cgh) {                                                    \
        cgh.parallel_for(                                                                         \
            ::sycl::nd_range<1>(global_range, local_range),                                       \
            PerTensorAbsMaxKernel<T, NVAL>(input_ptr, output_s_ptr, num_elements));               \
      });                                                                                         \
      queue.wait_and_throw();                                                                     \
    }                                                                                             \
    queue.submit([&](::sycl::handler& cgh) {                                                      \
      cgh.parallel_for(                                                                           \
          ::sycl::nd_range<1>(global_range, local_range),                                         \
          PerTensorQuantFP8Kernel<T, NVAL>(input_ptr, output_q_ptr, output_s_ptr, num_elements)); \
    });                                                                                           \
  } while (0)

  switch (vec_size) {
    case 16:
      SGL_PTQ_LAUNCH(16);
      break;
    case 8:
      SGL_PTQ_LAUNCH(8);
      break;
    case 4:
      SGL_PTQ_LAUNCH(4);
      break;
    case 2:
      SGL_PTQ_LAUNCH(2);
      break;
    default:
      SGL_PTQ_LAUNCH(1);
      break;
  }
#undef SGL_PTQ_LAUNCH
  // NOTE: the dynamic path waits between absmax and quantize (the quantize reads
  // the scalar the absmax pass wrote); no trailing .wait() -- PyTorch owns final sync.
}

}  // namespace per_tensor_quant_fp8

// ---------------------------------------------------------------------------
// C API for the Python ctypes wrapper. One symbol per (dtype, is_static).
// ---------------------------------------------------------------------------
#define _DEFINE_PTQ_FORWARD(DTYPE_SUFFIX, DTYPE, STATIC_SUFFIX, IS_STATIC)                        \
  extern "C" void per_tensor_quant_fp8_forward_##DTYPE_SUFFIX##_##STATIC_SUFFIX(                  \
      void* queue_ptr, const void* input, void* output_q, void* output_s, int64_t num_elements) { \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr);                                        \
    per_tensor_quant_fp8::per_tensor_quant_fp8_launcher<DTYPE, IS_STATIC>(                        \
        queue, input, output_q, output_s, num_elements);                                          \
  }
#define DEFINE_PTQ_FORWARD(DTYPE_SUFFIX, DTYPE, STATIC_SUFFIX, IS_STATIC) \
  _DEFINE_PTQ_FORWARD(DTYPE_SUFFIX, DTYPE, STATIC_SUFFIX, IS_STATIC)

#define DEFINE_PTQ_BOTH_STATIC(DTYPE_SUFFIX, DTYPE)       \
  DEFINE_PTQ_FORWARD(DTYPE_SUFFIX, DTYPE, dynamic, false) \
  DEFINE_PTQ_FORWARD(DTYPE_SUFFIX, DTYPE, static, true)

// Compile only the requested (dtype, is_static) when the selectors are defined,
// else emit all combinations (for testing / pre-compilation). AOT supports only
// fp16/bf16 inputs; fp8 output is always e4m3fn.
#if defined(SGL_PTQ_DTYPE_fp16) || defined(SGL_PTQ_DTYPE_bf16)
#if defined(SGL_PTQ_DTYPE_fp16)
#define _PTQ_SUFFIX fp16
#define _PTQ_T ::sycl::half
#else
#define _PTQ_SUFFIX bf16
#define _PTQ_T ::sycl::ext::oneapi::bfloat16
#endif
#if defined(SGL_PTQ_STATIC_true)
DEFINE_PTQ_FORWARD(_PTQ_SUFFIX, _PTQ_T, static, true)
#elif defined(SGL_PTQ_STATIC_false)
DEFINE_PTQ_FORWARD(_PTQ_SUFFIX, _PTQ_T, dynamic, false)
#else
DEFINE_PTQ_BOTH_STATIC(_PTQ_SUFFIX, _PTQ_T)
#endif
#else
DEFINE_PTQ_BOTH_STATIC(fp16, ::sycl::half)
DEFINE_PTQ_BOTH_STATIC(bf16, ::sycl::ext::oneapi::bfloat16)
#endif

#undef DEFINE_PTQ_BOTH_STATIC
#undef DEFINE_PTQ_FORWARD
#undef _DEFINE_PTQ_FORWARD

}  // namespace sycl_kernel
}  // namespace sgl
