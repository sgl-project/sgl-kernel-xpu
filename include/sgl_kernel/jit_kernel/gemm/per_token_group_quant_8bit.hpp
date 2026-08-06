/**
 * Per-Token-Group 8-bit Quantization SYCL JIT Kernel for SGLang XPU
 *
 * Quantizes a contiguous input tensor in fixed-size groups along the last dim
 * to int8 or fp8 (e4m3fn), emitting one fp32 (or UE8M0-packed) scale per group:
 *
 *   for each group g of GROUP_SIZE elements:
 *     absmax = max(eps, max|x_i|)
 *     y_s    = absmax / max_8bit
 *     x_q_i  = clamp(x_i / y_s, min_8bit, max_8bit)   -> int8 / e4m3fn
 *
 * Self-contained JIT port of the AOT kernel in
 * src/sycl/per_token_group_quant_8bit.cpp. Kept numerically close to it:
 *   - 32-wide sub-group holding two 16-thread quantization groups
 *   - XOR butterfly absmax reduce within each 16-thread group
 *   - vectorized load/store (VEC_SIZE = 16 / sizeof(T))
 *   - fp32 absmax / scale accumulation
 *   - one scale written per group by lane 0
 *
 * Differences from AOT (kept behaviorally identical):
 *   - FP8 e4m3fn conversion uses a self-contained c10-style routine instead of
 *     CUTLASS (bit-exact with torch's `.to(torch.float8_e4m3fn)` in the clamped
 *     [-448, 448] range this kernel produces).
 *   - GROUP_SIZE and the (input, output) dtypes are compile-time specialized via
 *     -D macros (one .so per config); is_column_major / scale_ue8m0 are runtime
 *     args (they only move where the single per-group scale is written).
 */

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

namespace sgl {
namespace sycl_kernel {

static constexpr int kPtgqThreadsPerGroup = 16;
static constexpr int kPtgqSubGroupSize = 32;

// c10-style round-to-nearest-even float -> e4m3fn (finite, NaN at 0x7f/0xff).
// Matches torch's fp8e4m3fn_from_fp32_value; inputs are pre-clamped to +-448.
inline uint8_t ptgq_fp32_to_e4m3fn(float f) {
  const uint32_t fp8_max = static_cast<uint32_t>(1087) << 20;
  const uint32_t denorm_mask = static_cast<uint32_t>(141) << 23;

  uint32_t f_bits = ::sycl::bit_cast<uint32_t>(f);
  uint8_t result = 0u;

  const uint32_t sign = f_bits & static_cast<uint32_t>(0x80000000);
  f_bits ^= sign;

  if (f_bits >= fp8_max) {
    // NaN / out-of-range -> saturate to NaN pattern (unreached after clamp).
    result = 0x7f;
  } else if (f_bits < (static_cast<uint32_t>(121) << 23)) {
    // Smaller than the smallest e4m3fn normal (2^-6): use denormal rounding.
    float shifted = ::sycl::bit_cast<float>(f_bits) + ::sycl::bit_cast<float>(denorm_mask);
    f_bits = ::sycl::bit_cast<uint32_t>(shifted);
    result = static_cast<uint8_t>(f_bits - denorm_mask);
  } else {
    // Normal: round-to-nearest-even.
    uint8_t mant_odd = (f_bits >> 20) & 1u;
    f_bits += (static_cast<uint32_t>(7 - 127) << 23) + 0x7FFFFu;
    f_bits += mant_odd;
    result = static_cast<uint8_t>(f_bits >> 20);
  }

  result |= static_cast<uint8_t>(sign >> 24);
  return result;
}

// XOR butterfly max-reduce within each 16-thread quantization group (the two
// halves of a 32-wide sub-group reduce independently since masks stay < 16).
template <typename T>
inline T ptgq_group_reduce_max(T val, ::sycl::nd_item<1> item) {
  auto sg = item.get_sub_group();
  val = ::sycl::fmax(val, ::sycl::permute_group_by_xor(sg, val, 8));
  val = ::sycl::fmax(val, ::sycl::permute_group_by_xor(sg, val, 4));
  val = ::sycl::fmax(val, ::sycl::permute_group_by_xor(sg, val, 2));
  val = ::sycl::fmax(val, ::sycl::permute_group_by_xor(sg, val, 1));
  return val;
}

// ---------------------------------------------------------------------------
// Kernel functor. Specialized on input dtype T, GROUP_SIZE and output type.
// ---------------------------------------------------------------------------
template <typename T, int GROUP_SIZE, bool kIsFp8>
class PerTokenGroupQuant8bitKernel {
 public:
  static constexpr int VEC_SIZE = 16 / sizeof(T);
  static constexpr int NUM_VEC_ELEMS = GROUP_SIZE / VEC_SIZE;
  static constexpr int VECS_PER_THREAD = (NUM_VEC_ELEMS + kPtgqThreadsPerGroup - 1) / kPtgqThreadsPerGroup;

  PerTokenGroupQuant8bitKernel(
      const T* input,
      void* output_q,
      void* output_s,
      int groups_per_block,
      float eps,
      float min_8bit,
      float max_8bit,
      bool scale_ue8m0,
      bool is_column_major,
      int num_groups_per_row,
      int scale_stride)
      : input_(input),
        output_q_(output_q),
        output_s_(output_s),
        groups_per_block_(groups_per_block),
        eps_(eps),
        min_8bit_(min_8bit),
        max_8bit_(max_8bit),
        scale_ue8m0_(scale_ue8m0),
        is_column_major_(is_column_major),
        num_groups_per_row_(num_groups_per_row),
        scale_stride_(scale_stride) {}

  [[sycl::reqd_sub_group_size(kPtgqSubGroupSize)]] void operator()(::sycl::nd_item<1> item) const {
    const int64_t local_group_id = item.get_local_id(0) / kPtgqThreadsPerGroup;
    const int lane_id = item.get_local_id(0) % kPtgqThreadsPerGroup;

    const int64_t block_group_id = item.get_group(0) * groups_per_block_;
    const int64_t global_group_id = block_group_id + local_group_id;
    const int64_t block_group_offset = global_group_id * GROUP_SIZE;

    const T* group_input = input_ + block_group_offset;
    uint8_t* group_output = static_cast<uint8_t*>(output_q_) + block_group_offset;

    using vec_type = ::sycl::vec<T, VEC_SIZE>;
    using float_vec_type = ::sycl::vec<float, VEC_SIZE>;

    vec_type input_vecs[VECS_PER_THREAD];
    float_vec_type input_vals[VECS_PER_THREAD];

    float local_absmax = eps_;

    // Single pass: load, convert to fp32, cache, and track absmax.
#pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; ++v) {
      const int i = lane_id + v * kPtgqThreadsPerGroup;
      if (i < NUM_VEC_ELEMS) {
        input_vecs[v].load(
            0, ::sycl::multi_ptr<const T, ::sycl::access::address_space::global_space>(group_input + i * VEC_SIZE));
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
          float val = static_cast<float>(input_vecs[v][j]);
          input_vals[v][j] = val;
          local_absmax = ::sycl::fmax(local_absmax, ::sycl::fabs(val));
        }
      }
    }

    local_absmax = ptgq_group_reduce_max(local_absmax, item);

    // Scale factor (optionally UE8M0-quantized to a power of two).
    float y_s = local_absmax / max_8bit_;
    if (scale_ue8m0_) {
      float exp_s = ::sycl::ceil(::sycl::log2(::sycl::fmax(y_s, 1e-10f)));
      y_s = ::sycl::exp2(exp_s);
      if (lane_id == 0) {
        // UE8M0: uint8 exponent (+127 bias), packed 4-per-uint32, column-major.
        const int num_elems_per_pack = 4;  // sizeof(uint32) / sizeof(uint8)
        const int row_idx = global_group_id / num_groups_per_row_;
        const int col_idx_unpacked = global_group_id % num_groups_per_row_;
        const int col_idx = col_idx_unpacked / num_elems_per_pack;
        const int pack_idx = col_idx_unpacked % num_elems_per_pack;
        uint8_t* scale_out = static_cast<uint8_t*>(output_s_) +
                             (col_idx * scale_stride_ * num_elems_per_pack + row_idx * num_elems_per_pack + pack_idx);
        *scale_out = static_cast<uint8_t>(static_cast<int>(exp_s) + 127);
      }
    } else if (lane_id == 0) {
      float* scale_out;
      if (is_column_major_) {
        const int row_idx = global_group_id / num_groups_per_row_;
        const int col_idx = global_group_id % num_groups_per_row_;
        scale_out = static_cast<float*>(output_s_) + (col_idx * scale_stride_ + row_idx);
      } else {
        scale_out = static_cast<float*>(output_s_) + global_group_id;
      }
      *scale_out = y_s;
    }

    const float inv_y_s = 1.0f / y_s;

    using out_vec_type = ::sycl::vec<uint8_t, VEC_SIZE>;
#pragma unroll
    for (int v = 0; v < VECS_PER_THREAD; ++v) {
      const int i = lane_id + v * kPtgqThreadsPerGroup;
      if (i < NUM_VEC_ELEMS) {
        out_vec_type out_vec;
#pragma unroll
        for (int j = 0; j < VEC_SIZE; ++j) {
          float q_val = ::sycl::fmin(::sycl::fmax(input_vals[v][j] * inv_y_s, min_8bit_), max_8bit_);
          if constexpr (kIsFp8) {
            out_vec[j] = ptgq_fp32_to_e4m3fn(q_val);
          } else {
            // int8: truncate toward zero, matching torch's `.to(int8)` and the
            // AOT `static_cast<int8_t>` path.
            out_vec[j] = ::sycl::bit_cast<uint8_t>(static_cast<int8_t>(q_val));
          }
        }
        out_vec.store(
            0,
            ::sycl::address_space_cast<::sycl::access::address_space::global_space, ::sycl::access::decorated::yes>(
                group_output + i * VEC_SIZE));
      }
    }
  }

 private:
  const T* input_;
  void* output_q_;
  void* output_s_;
  int groups_per_block_;
  float eps_;
  float min_8bit_;
  float max_8bit_;
  bool scale_ue8m0_;
  bool is_column_major_;
  int num_groups_per_row_;
  int scale_stride_;
};

// ---------------------------------------------------------------------------
// Host launcher.
// ---------------------------------------------------------------------------
template <typename T, int GROUP_SIZE, bool kIsFp8>
void per_token_group_quant_8bit_launcher(
    ::sycl::queue& queue,
    const void* input,
    void* output_q,
    void* output_s,
    int64_t num_groups,
    int64_t groups_per_block,
    float eps,
    float min_8bit,
    float max_8bit,
    bool scale_ue8m0,
    bool is_column_major,
    int64_t num_groups_per_row,
    int64_t scale_stride) {
  if (num_groups <= 0) return;

  const T* input_ptr = static_cast<const T*>(input);
  const int64_t num_blocks = num_groups / groups_per_block;
  const int64_t num_threads = groups_per_block * kPtgqThreadsPerGroup;

  PerTokenGroupQuant8bitKernel<T, GROUP_SIZE, kIsFp8> kernel(
      input_ptr,
      output_q,
      output_s,
      static_cast<int>(groups_per_block),
      eps,
      min_8bit,
      max_8bit,
      scale_ue8m0,
      is_column_major,
      static_cast<int>(num_groups_per_row),
      static_cast<int>(scale_stride));

  queue.submit([&](::sycl::handler& cgh) {
    cgh.parallel_for(
        ::sycl::nd_range<1>(::sycl::range<1>(num_blocks * num_threads), ::sycl::range<1>(num_threads)), kernel);
  });
  // NOTE: no .wait() -- PyTorch owns stream synchronization.
}

// ---------------------------------------------------------------------------
// C API for the Python ctypes wrapper. One symbol per (in dtype, out dtype,
// group size); is_column_major / scale_ue8m0 are runtime flags.
// ---------------------------------------------------------------------------
#define _DEFINE_PTGQ_FORWARD(IN_SUFFIX, IN_T, OUT_SUFFIX, IS_FP8, GS)                   \
  extern "C" void per_token_group_quant_8bit_forward_##IN_SUFFIX##_##OUT_SUFFIX##_##GS( \
      void* queue_ptr,                                                                  \
      const void* input,                                                                \
      void* output_q,                                                                   \
      void* output_s,                                                                   \
      int64_t num_groups,                                                               \
      int64_t groups_per_block,                                                         \
      float eps,                                                                        \
      float min_8bit,                                                                   \
      float max_8bit,                                                                   \
      int32_t scale_ue8m0,                                                              \
      int32_t is_column_major,                                                          \
      int64_t num_groups_per_row,                                                       \
      int64_t scale_stride) {                                                           \
    auto& queue = *static_cast<::sycl::queue*>(queue_ptr);                              \
    per_token_group_quant_8bit_launcher<IN_T, GS, IS_FP8>(                              \
        queue,                                                                          \
        input,                                                                          \
        output_q,                                                                       \
        output_s,                                                                       \
        num_groups,                                                                     \
        groups_per_block,                                                               \
        eps,                                                                            \
        min_8bit,                                                                       \
        max_8bit,                                                                       \
        scale_ue8m0 != 0,                                                               \
        is_column_major != 0,                                                           \
        num_groups_per_row,                                                             \
        scale_stride);                                                                  \
  }
#define DEFINE_PTGQ_FORWARD(IN_SUFFIX, IN_T, OUT_SUFFIX, IS_FP8, GS) \
  _DEFINE_PTGQ_FORWARD(IN_SUFFIX, IN_T, OUT_SUFFIX, IS_FP8, GS)

#define DEFINE_PTGQ_IN_OUT(GS)                                              \
  DEFINE_PTGQ_FORWARD(fp32, float, int8, false, GS)                         \
  DEFINE_PTGQ_FORWARD(fp32, float, fp8, true, GS)                           \
  DEFINE_PTGQ_FORWARD(fp16, ::sycl::half, int8, false, GS)                  \
  DEFINE_PTGQ_FORWARD(fp16, ::sycl::half, fp8, true, GS)                    \
  DEFINE_PTGQ_FORWARD(bf16, ::sycl::ext::oneapi::bfloat16, int8, false, GS) \
  DEFINE_PTGQ_FORWARD(bf16, ::sycl::ext::oneapi::bfloat16, fp8, true, GS)

// When SGL_PTGQ_GROUP_SIZE (+ optional dtype selectors) is defined, compile only
// the requested variant. Otherwise emit all configs (testing / pre-compilation).
#if defined(SGL_PTGQ_GROUP_SIZE)
#if defined(SGL_PTGQ_IN_fp32)
#define _PTGQ_IN_SUFFIX fp32
#define _PTGQ_IN_T float
#elif defined(SGL_PTGQ_IN_fp16)
#define _PTGQ_IN_SUFFIX fp16
#define _PTGQ_IN_T ::sycl::half
#elif defined(SGL_PTGQ_IN_bf16)
#define _PTGQ_IN_SUFFIX bf16
#define _PTGQ_IN_T ::sycl::ext::oneapi::bfloat16
#endif

#if defined(_PTGQ_IN_SUFFIX)
#if defined(SGL_PTGQ_OUT_fp8)
DEFINE_PTGQ_FORWARD(_PTGQ_IN_SUFFIX, _PTGQ_IN_T, fp8, true, SGL_PTGQ_GROUP_SIZE)
#elif defined(SGL_PTGQ_OUT_int8)
DEFINE_PTGQ_FORWARD(_PTGQ_IN_SUFFIX, _PTGQ_IN_T, int8, false, SGL_PTGQ_GROUP_SIZE)
#else
DEFINE_PTGQ_FORWARD(_PTGQ_IN_SUFFIX, _PTGQ_IN_T, fp8, true, SGL_PTGQ_GROUP_SIZE)
DEFINE_PTGQ_FORWARD(_PTGQ_IN_SUFFIX, _PTGQ_IN_T, int8, false, SGL_PTGQ_GROUP_SIZE)
#endif
#else
DEFINE_PTGQ_IN_OUT(SGL_PTGQ_GROUP_SIZE)
#endif
#else
DEFINE_PTGQ_IN_OUT(32)
DEFINE_PTGQ_IN_OUT(64)
DEFINE_PTGQ_IN_OUT(128)
DEFINE_PTGQ_IN_OUT(256)
DEFINE_PTGQ_IN_OUT(512)
#endif

#undef DEFINE_PTGQ_IN_OUT
#undef DEFINE_PTGQ_FORWARD
#undef _DEFINE_PTGQ_FORWARD

}  // namespace sycl_kernel
}  // namespace sgl
