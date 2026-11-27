// Per-token (per-row) dynamic symmetric INT8 quantization for XPU.
// SYCL equivalent of sglang's Triton _per_token_quant_int8 kernel.
//
// For each row of an [..., hidden_dim] tensor:
//   absmax = max(rowmax(|x|), ABSMAX_EPS)
//   scale  = absmax / INT8_QUANT_MAX
//   q[i]   = round(x[i] * (INT8_QUANT_MAX / absmax))  as int8
// Optional output_sum writes rowsum(x) per token when CAL_SUM is enabled.

#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/core/DeviceGuard.h>
#include <c10/xpu/XPUStream.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <sycl/sycl.hpp>
#include <type_traits>

#include "MemoryAccess.h"
#include "SGLKernelPerf.h"
#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

namespace {

constexpr int SUB_GROUP_SIZE = 32;
constexpr int TOKENS_PER_WG = 8;  // sub-groups (tokens) per work-group in the warp kernel
constexpr int STREAM_BLOCK_WG_SIZE = 512;
constexpr int CACHED_VEC_SIZE = 4;
constexpr float INT8_QUANT_MAX = 127.0f;
constexpr float ABSMAX_EPS = 1e-10f;

template <typename Int>
constexpr Int ceil_div(Int a, Int b) {
  return (a + b - 1) / b;
}

enum class ScaleDtype : int {
  FP32 = 0,
  FP16 = 1,
  BF16 = 2,
};

inline ScaleDtype to_scale_dtype(at::ScalarType st) {
  switch (st) {
    case at::kFloat:
      return ScaleDtype::FP32;
    case at::kHalf:
      return ScaleDtype::FP16;
    case at::kBFloat16:
      return ScaleDtype::BF16;
    default:
      TORCH_CHECK(false, "Unsupported scale dtype: ", st);
  }
}

inline void write_scale(void* output_s, int64_t token_id, float scale, ScaleDtype dtype) {
  switch (dtype) {
    case ScaleDtype::FP32:
      static_cast<float*>(output_s)[token_id] = scale;
      break;
    case ScaleDtype::FP16:
      static_cast<sycl::half*>(output_s)[token_id] = static_cast<sycl::half>(scale);
      break;
    case ScaleDtype::BF16:
      static_cast<sycl::ext::oneapi::bfloat16*>(output_s)[token_id] = static_cast<sycl::ext::oneapi::bfloat16>(scale);
      break;
  }
}

// Shared POD bundle for all quantization kernels.
template <typename T>
struct QuantParams {
  const T* input;
  int8_t* output_q;
  void* output_s;
  T* output_sum;
  int64_t hidden_dim;
  int64_t num_tokens;
  ScaleDtype scale_dtype;
};

// ---------------------------------------------------------------------------
// Coalesced 4-element vector helper: emits a single LSC vector load
// (d32x2 for fp16/bf16, d32x4 for fp32) and a single 32-bit int8x4 store
// (d32) per lane.
// ---------------------------------------------------------------------------
template <typename T>
struct RawPack4 {
  static_assert(sizeof(T) == 2 || sizeof(T) == 4, "Unsupported input element size");
  using storage_t = std::conditional_t<sizeof(T) == 2, sycl::uint2, sycl::float4>;

  struct Elem4 {
    T v[4];
  };
  struct Int8x4 {
    int8_t v[4];
  };

  static inline storage_t load(const T* ptr) {
    return *reinterpret_cast<const storage_t*>(ptr);
  }

  static inline Elem4 unpack(storage_t raw) {
    return sycl::bit_cast<Elem4>(raw);
  }

  static inline uint32_t quantize_pack(storage_t raw, float scale_inv) {
    const Elem4 e = unpack(raw);
    Int8x4 q;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      // sycl::round, unlike rint, rounds half away from zero like the CUDA op.
      q.v[j] = static_cast<int8_t>(sycl::round(static_cast<float>(e.v[j]) * scale_inv));
    }
    return sycl::bit_cast<uint32_t>(q);
  }

  static inline void store(int8_t* ptr, uint32_t packed_q) {
    *reinterpret_cast<uint32_t*>(ptr) = packed_q;
  }
};

// ---------------------------------------------------------------------------
// GRF register buffer shared by the single-pass cached Warp and Block kernels.
// Keeps the exact load -> reduce -> quantize -> store loop structure so the
// compiler schedules LSC vector ops and GRF allocation identically.
// ---------------------------------------------------------------------------
template <typename T, int ITERS>
struct CachedRowVecs {
  using Pack = RawPack4<T>;
  using storage_t = typename Pack::storage_t;

  storage_t regs[ITERS]{};

  inline void load(const T* token_input, uint32_t lane_offset, uint32_t stride, uint32_t num_vecs) {
    if constexpr (ITERS > 1) {
#pragma unroll
      for (int it = 0; it < ITERS - 1; ++it) {
        const uint32_t vec_idx = static_cast<uint32_t>(it) * stride + lane_offset;
        regs[it] = Pack::load(token_input + vec_idx * CACHED_VEC_SIZE);
      }
    }
    const uint32_t last_vec_idx = static_cast<uint32_t>(ITERS - 1) * stride + lane_offset;
    if (last_vec_idx < num_vecs) {
      regs[ITERS - 1] = Pack::load(token_input + last_vec_idx * CACHED_VEC_SIZE);
    }
  }

  template <bool CAL_SUM>
  inline void reduce_local(float& max_value, float& sum_value) const {
    max_value = 0.0f;
    sum_value = 0.0f;
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const auto elems = Pack::unpack(regs[it]);
#pragma unroll
      for (int j = 0; j < CACHED_VEC_SIZE; ++j) {
        const float val = static_cast<float>(elems.v[j]);
        max_value = sycl::fmax(max_value, sycl::fabs(val));
        if constexpr (CAL_SUM) {
          sum_value += val;
        }
      }
    }
  }

  inline void quantize_and_store(
      int8_t* token_output, float scale_inv, uint32_t lane_offset, uint32_t stride, uint32_t num_vecs) const {
    uint32_t q_packed[ITERS];
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      q_packed[it] = Pack::quantize_pack(regs[it], scale_inv);
    }

    if constexpr (ITERS > 1) {
#pragma unroll
      for (int it = 0; it < ITERS - 1; ++it) {
        const uint32_t vec_idx = static_cast<uint32_t>(it) * stride + lane_offset;
        Pack::store(token_output + vec_idx * CACHED_VEC_SIZE, q_packed[it]);
      }
    }
    const uint32_t last_vec_idx = static_cast<uint32_t>(ITERS - 1) * stride + lane_offset;
    if (last_vec_idx < num_vecs) {
      Pack::store(token_output + last_vec_idx * CACHED_VEC_SIZE, q_packed[ITERS - 1]);
    }
  }
};

// ---------------------------------------------------------------------------
// Single-barrier work-group reduction over SLM for absmax (and optional sum).
// Writes per-token scale (and sum) from lane 0 of sub-group 0, and returns
// the clamped absmax broadcast to all lanes in the work-group.
// ---------------------------------------------------------------------------
template <typename T, int WG_SIZE, bool CAL_SUM>
inline float reduce_wg_and_write_meta(
    sycl::nd_item<1> item,
    float* slm,
    float max_value,
    float sum_value,
    int64_t token_id,
    const QuantParams<T>& params) {
  constexpr int NUM_SGS = WG_SIZE / SUB_GROUP_SIZE;
  static_assert(WG_SIZE % SUB_GROUP_SIZE == 0 && NUM_SGS <= SUB_GROUP_SIZE);

  auto sg = item.get_sub_group();
  const uint32_t tid = static_cast<uint32_t>(item.get_local_id(0));
  const uint32_t sg_id = tid / SUB_GROUP_SIZE;
  const uint32_t lane_id = tid % SUB_GROUP_SIZE;

  max_value = sycl::reduce_over_group(sg, max_value, sycl::maximum<float>());
  if constexpr (CAL_SUM) {
    sum_value = sycl::reduce_over_group(sg, sum_value, sycl::plus<float>());
  }

  if (lane_id == 0) {
    slm[sg_id] = max_value;
    if constexpr (CAL_SUM) {
      slm[NUM_SGS + sg_id] = sum_value;
    }
  }
  sycl::group_barrier(item.get_group());

  const float sg_max = (lane_id < NUM_SGS) ? slm[lane_id] : 0.0f;
  max_value = sycl::fmax(sycl::reduce_over_group(sg, sg_max, sycl::maximum<float>()), ABSMAX_EPS);

  if constexpr (CAL_SUM) {
    if (sg_id == 0) {
      const float sg_sum = (lane_id < NUM_SGS) ? slm[NUM_SGS + lane_id] : 0.0f;
      sum_value = sycl::reduce_over_group(sg, sg_sum, sycl::plus<float>());
      if (lane_id == 0) {
        params.output_sum[token_id] = static_cast<T>(sum_value);
      }
    }
  }

  if (tid == 0) {
    write_scale(params.output_s, token_id, max_value / INT8_QUANT_MAX, params.scale_dtype);
  }
  return max_value;
}

// ---------------------------------------------------------------------------
// Kernel 1: Single-pass register-cached warp kernel.
// One sub-group (32 lanes) handles one token with zero barriers and zero SLM
// when hidden_dim is small (<=1024) and num_tokens saturates all Xe-cores.
// ---------------------------------------------------------------------------
template <typename T, int ITERS, bool CAL_SUM>
class PerTokenQuantInt8CachedWarpKernel {
 private:
  QuantParams<T> params_;

 public:
  explicit PerTokenQuantInt8CachedWarpKernel(QuantParams<T> params) : params_(params) {}

  [[sycl::reqd_work_group_size(TOKENS_PER_WG * SUB_GROUP_SIZE)]] [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]]
  void operator()(sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    const uint32_t tid = static_cast<uint32_t>(item.get_local_id(0));
    const uint32_t warp_id = tid / SUB_GROUP_SIZE;
    const uint32_t lane_id = tid % SUB_GROUP_SIZE;
    const int64_t token_id = static_cast<int64_t>(item.get_group(0)) * TOKENS_PER_WG + warp_id;
    if (token_id >= params_.num_tokens) return;

    const uint32_t hidden_dim = static_cast<uint32_t>(params_.hidden_dim);
    const T* token_input = params_.input + token_id * params_.hidden_dim;
    int8_t* token_output = params_.output_q + token_id * params_.hidden_dim;
    const uint32_t num_vecs = hidden_dim / CACHED_VEC_SIZE;

    CachedRowVecs<T, ITERS> row;
    row.load(token_input, lane_id, SUB_GROUP_SIZE, num_vecs);

    float max_value = 0.0f;
    float sum_value = 0.0f;
    row.template reduce_local<CAL_SUM>(max_value, sum_value);

    max_value = sycl::fmax(sycl::reduce_over_group(sg, max_value, sycl::maximum<float>()), ABSMAX_EPS);
    if constexpr (CAL_SUM) {
      sum_value = sycl::reduce_over_group(sg, sum_value, sycl::plus<float>());
      if (lane_id == 0) {
        params_.output_sum[token_id] = static_cast<T>(sum_value);
      }
    }
    if (lane_id == 0) {
      write_scale(params_.output_s, token_id, max_value / INT8_QUANT_MAX, params_.scale_dtype);
    }

    const float scale_inv = INT8_QUANT_MAX / max_value;
    row.quantize_and_store(token_output, scale_inv, lane_id, SUB_GROUP_SIZE, num_vecs);
  }
};

// ---------------------------------------------------------------------------
// Kernel 2: Single-pass register-cached block kernel.
// One work-group (WG_SIZE lanes) handles one token, keeping the row in GRF
// registers across a single-barrier SLM reduction so global memory is read
// only once (avoids LLC thrashing in Long Prefill and Pass-2 L2 reloads).
// ---------------------------------------------------------------------------
template <typename T, int WG_SIZE, int ITERS, bool CAL_SUM>
class PerTokenQuantInt8CachedBlockKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int NUM_SGS = WG_SIZE / SUB_GROUP_SIZE;

 private:
  QuantParams<T> params_;
  sycl::local_accessor<float, 1> slm_{};

 public:
  explicit PerTokenQuantInt8CachedBlockKernel(QuantParams<T> params) : params_(params) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    constexpr int slm_floats = (CAL_SUM ? 2 : 1) * NUM_SGS;
    slm_ = sycl::local_accessor<float, 1>(slm_floats, cgh);
  }

  [[sycl::reqd_work_group_size(WG_SIZE)]] [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]]
  void operator()(sycl::nd_item<1> item) const {
    const int64_t token_id = static_cast<int64_t>(item.get_group(0));
    if (token_id >= params_.num_tokens) return;

    const uint32_t tid = static_cast<uint32_t>(item.get_local_id(0));
    const uint32_t hidden_dim = static_cast<uint32_t>(params_.hidden_dim);
    const T* token_input = params_.input + token_id * params_.hidden_dim;
    int8_t* token_output = params_.output_q + token_id * params_.hidden_dim;
    const uint32_t num_vecs = hidden_dim / CACHED_VEC_SIZE;

    CachedRowVecs<T, ITERS> row;
    row.load(token_input, tid, WG_SIZE, num_vecs);

    float max_value = 0.0f;
    float sum_value = 0.0f;
    row.template reduce_local<CAL_SUM>(max_value, sum_value);

    float* slm = slm_.template get_multi_ptr<sycl::access::decorated::no>().get();
    max_value = reduce_wg_and_write_meta<T, WG_SIZE, CAL_SUM>(item, slm, max_value, sum_value, token_id, params_);

    const float scale_inv = INT8_QUANT_MAX / max_value;
    row.quantize_and_store(token_output, scale_inv, tid, WG_SIZE, num_vecs);
  }
};

// ---------------------------------------------------------------------------
// Kernel 3: Two-pass fallback block kernel.
// Handles unaligned pointers, non-4-divisible hidden_dim, or extreme
// hidden_dim (>32768). One work-group per token keeps active rows in LLC.
// ---------------------------------------------------------------------------
template <typename T, int WG_SIZE, int VEC_SIZE, bool CAL_SUM>
class PerTokenQuantInt8StreamBlockKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int NUM_SGS = WG_SIZE / SUB_GROUP_SIZE;

 private:
  QuantParams<T> params_;
  sycl::local_accessor<float, 1> slm_{};

 public:
  explicit PerTokenQuantInt8StreamBlockKernel(QuantParams<T> params) : params_(params) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    constexpr int slm_floats = (CAL_SUM ? 2 : 1) * NUM_SGS;
    slm_ = sycl::local_accessor<float, 1>(slm_floats, cgh);
  }

  [[sycl::reqd_work_group_size(WG_SIZE)]] [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]]
  void operator()(sycl::nd_item<1> item) const {
    const int64_t token_id = static_cast<int64_t>(item.get_group(0));
    if (token_id >= params_.num_tokens) return;

    const uint32_t tid = static_cast<uint32_t>(item.get_local_id(0));
    const T* token_input = params_.input + token_id * params_.hidden_dim;
    int8_t* token_output = params_.output_q + token_id * params_.hidden_dim;
    const int64_t num_vec_elems = params_.hidden_dim / VEC_SIZE;

    using vec_in_t = vec_t<T, VEC_SIZE>;
    using vec_out_t = vec_t<int8_t, VEC_SIZE>;

    // Pass 1: find row absmax (and optional sum).
    float max_value = 0.0f;
    float sum_value = 0.0f;
    for (int64_t i = tid; i < num_vec_elems; i += WG_SIZE) {
      vec_in_t in;
      in.load(0, sycl::multi_ptr<const T, sycl::access::address_space::global_space>(token_input + i * VEC_SIZE));
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const float val = static_cast<float>(in[j]);
        max_value = sycl::fmax(max_value, sycl::fabs(val));
        if constexpr (CAL_SUM) {
          sum_value += val;
        }
      }
    }

    float* slm = slm_.template get_multi_ptr<sycl::access::decorated::no>().get();
    max_value = reduce_wg_and_write_meta<T, WG_SIZE, CAL_SUM>(item, slm, max_value, sum_value, token_id, params_);

    // Pass 2: quantize and store.
    const float scale_inv = INT8_QUANT_MAX / max_value;
    for (int64_t i = tid; i < num_vec_elems; i += WG_SIZE) {
      vec_in_t in;
      in.load(0, sycl::multi_ptr<const T, sycl::access::address_space::global_space>(token_input + i * VEC_SIZE));
      vec_out_t out;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        // sycl::round, unlike rint, rounds half away from zero like the CUDA op.
        out[j] = static_cast<int8_t>(sycl::round(static_cast<float>(in[j]) * scale_inv));
      }
      out.store(
          0,
          sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::yes>(
              token_output + i * VEC_SIZE));
    }
  }
};

// Compile-time dispatch for ITERS in [1, MaxIters] without instantiating
// unused register-budget tiers (e.g. ITERS > 4 for FP32).
template <int MaxIters, typename Fn>
void dispatch_iters(int iters, Fn&& fn) {
  static_assert(MaxIters >= 1 && MaxIters <= 8);
  const int clamped = (iters <= 1) ? 1 : iters;
  switch (clamped) {
    case 1:
      fn(std::integral_constant<int, 1>{});
      return;
    case 2:
      if constexpr (MaxIters >= 2) {
        fn(std::integral_constant<int, 2>{});
        return;
      }
      break;
    case 3:
      if constexpr (MaxIters >= 3) {
        fn(std::integral_constant<int, 3>{});
        return;
      }
      break;
    case 4:
      if constexpr (MaxIters >= 4) {
        fn(std::integral_constant<int, 4>{});
        return;
      }
      break;
    case 5:
      if constexpr (MaxIters >= 5) {
        fn(std::integral_constant<int, 5>{});
        return;
      }
      break;
    case 6:
      if constexpr (MaxIters >= 6) {
        fn(std::integral_constant<int, 6>{});
        return;
      }
      break;
    case 7:
      if constexpr (MaxIters >= 7) {
        fn(std::integral_constant<int, 7>{});
        return;
      }
      break;
    case 8:
      if constexpr (MaxIters >= 8) {
        fn(std::integral_constant<int, 8>{});
        return;
      }
      break;
    default:
      break;
  }
  TORCH_CHECK(false, "Internal error: invalid cached iters=", iters, " (max=", MaxIters, ")");
}

template <typename T, int WG_SIZE, int MaxIters, bool CAL_SUM>
void launch_cached_block(sycl::queue& Q, const QuantParams<T>& params, int64_t num_vecs) {
  const int iters = static_cast<int>(ceil_div<int64_t>(num_vecs, WG_SIZE));
  dispatch_iters<MaxIters>(iters, [&](auto iters_tag) {
    constexpr int ITERS = decltype(iters_tag)::value;
    sycl::range<1> global_range(static_cast<size_t>(params.num_tokens) * WG_SIZE);
    sycl::range<1> local_range(WG_SIZE);
    auto kernel = PerTokenQuantInt8CachedBlockKernel<T, WG_SIZE, ITERS, CAL_SUM>(params);
    sycl_kernel_submit(global_range, local_range, Q, kernel);
  });
}

template <typename T, bool CAL_SUM>
bool try_launch_cached_kernel(sycl::queue& Q, const QuantParams<T>& params, int64_t xe_cores, int64_t max_wg) {
  // 16-bit elements use half the GRF footprint of FP32, allowing deeper caching
  // per lane before risking register spilling.
  constexpr bool IS_16BIT = (sizeof(T) == 2);
  constexpr int MAX_WARP_ITERS = IS_16BIT ? 8 : 4;
  constexpr int MAX_NARROW_WG_ITERS = IS_16BIT ? 7 : 4;
  constexpr int MAX_WIDE_WG_ITERS = IS_16BIT ? 8 : 4;

  const int64_t num_tokens = params.num_tokens;
  const int64_t num_vecs = params.hidden_dim / CACHED_VEC_SIZE;

  // Tier 1 (Warp): small rows + large batch -> 1 sub-group/token, 0 barriers, 0 SLM.
  const int64_t min_tokens_for_warp = xe_cores * 2 * TOKENS_PER_WG;
  if (num_tokens >= min_tokens_for_warp && num_vecs <= SUB_GROUP_SIZE * MAX_WARP_ITERS) {
    const int iters = static_cast<int>(ceil_div<int64_t>(num_vecs, SUB_GROUP_SIZE));
    dispatch_iters<MAX_WARP_ITERS>(iters, [&](auto iters_tag) {
      constexpr int ITERS = decltype(iters_tag)::value;
      constexpr int WG_SIZE = TOKENS_PER_WG * SUB_GROUP_SIZE;
      const int64_t num_wgs = ceil_div<int64_t>(num_tokens, TOKENS_PER_WG);
      sycl::range<1> global_range(static_cast<size_t>(num_wgs) * WG_SIZE);
      sycl::range<1> local_range(WG_SIZE);
      auto kernel = PerTokenQuantInt8CachedWarpKernel<T, ITERS, CAL_SUM>(params);
      sycl_kernel_submit(global_range, local_range, Q, kernel);
    });
    return true;
  }

  // Tier 2 (WG=128): narrow rows; skip when batch is tiny so each token gets
  // enough threads to hide memory latency.
  if (num_vecs <= 128 * MAX_NARROW_WG_ITERS && !(num_tokens < xe_cores && num_vecs > 128)) {
    launch_cached_block<T, 128, MAX_NARROW_WG_ITERS, CAL_SUM>(Q, params, num_vecs);
    return true;
  }

  // Tier 3 (WG=256): medium rows; step up to WG=512 for small-batch wide rows.
  if (num_vecs <= 256 * MAX_NARROW_WG_ITERS && !(num_tokens <= 32 && num_vecs > 512)) {
    launch_cached_block<T, 256, MAX_NARROW_WG_ITERS, CAL_SUM>(Q, params, num_vecs);
    return true;
  }

  // Tier 4 (WG=512): wide rows.
  if (num_vecs <= 512 * MAX_WIDE_WG_ITERS) {
    launch_cached_block<T, 512, MAX_WIDE_WG_ITERS, CAL_SUM>(Q, params, num_vecs);
    return true;
  }

  // Tier 5 (WG=1024): extra-wide 16-bit rows (up to hidden_dim = 32768).
  if constexpr (IS_16BIT) {
    if (max_wg >= 1024 && num_vecs <= 1024 * MAX_WIDE_WG_ITERS) {
      launch_cached_block<T, 1024, MAX_WIDE_WG_ITERS, CAL_SUM>(Q, params, num_vecs);
      return true;
    }
  }

  return false;
}

int compute_max_vec_size(const at::Tensor& input, const at::Tensor& output_q, int64_t hidden_dim) {
  const uintptr_t in_addr = reinterpret_cast<uintptr_t>(input.data_ptr());
  const uintptr_t out_addr = reinterpret_cast<uintptr_t>(output_q.data_ptr());
  const size_t elem_size = static_cast<size_t>(input.element_size());

  int vec_size = CACHED_VEC_SIZE;
  while (vec_size > 1 && ((hidden_dim % vec_size != 0) || (in_addr % (vec_size * elem_size) != 0) ||
                          (out_addr % static_cast<size_t>(vec_size) != 0))) {
    vec_size /= 2;
  }
  return vec_size;
}

void validate_inputs(
    const at::Tensor& input,
    const at::Tensor& output_q,
    const at::Tensor& output_s,
    const std::optional<at::Tensor>& output_sum) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);
  TORCH_CHECK(input.dim() >= 1 && input.size(-1) > 0, "input must have a nonempty last dimension");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kFloat,
      "input must be Float16, BFloat16, or Float32");
  TORCH_CHECK(output_q.scalar_type() == at::kChar, "output_q must be Int8");
  TORCH_CHECK(output_q.sizes() == input.sizes(), "output_q must have the same shape as input");
  TORCH_CHECK(
      output_s.scalar_type() == at::kHalf || output_s.scalar_type() == at::kBFloat16 ||
          output_s.scalar_type() == at::kFloat,
      "output_s must be Float16, BFloat16, or Float32");

  if (output_sum.has_value()) {
    const auto& sums = *output_sum;
    CHECK_INPUT(sums);
    TORCH_CHECK(sums.device() == input.device(), "output_sum must be on the same device as input");
    TORCH_CHECK(sums.scalar_type() == input.scalar_type(), "output_sum must have the same dtype as input");
    auto sum_shape = input.sizes().vec();
    sum_shape.pop_back();
    TORCH_CHECK(sums.sizes() == at::IntArrayRef(sum_shape), "output_sum must have shape input.shape[:-1]");
  }
}

}  // namespace

SGL_KERNEL_EXPORT void sgl_per_token_quant_int8(
    at::Tensor input, at::Tensor output_q, at::Tensor output_s, const std::optional<at::Tensor>& output_sum) {
  validate_inputs(input, output_q, output_s, output_sum);

  const int64_t hidden_dim = input.size(-1);
  const int64_t num_tokens = input.numel() / hidden_dim;
  if (num_tokens == 0) return;

  const c10::OptionalDeviceGuard guard(input.device());
  auto& Q = dpcppGetCurrentQueue();
  const DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const ScaleDtype scale_dtype = to_scale_dtype(output_s.scalar_type());
  const int vec_size = compute_max_vec_size(input, output_q, hidden_dim);
  const int64_t xe_cores = dpcppMaxComputeUnitSize(dev_id);
  const int64_t max_wg = dpcppMaxWorkGroupSize(dev_id);
  const bool cal_sum = output_sum.has_value();

  AT_DISPATCH_FLOATING_TYPES_V2(input.scalar_type(), "sgl_per_token_quant_int8_input", [&] {
    using input_t = sycl_type_t<scalar_t>;
    SGL_DISPATCH_BOOL(cal_sum, CAL_SUM_V, [&] {
      const QuantParams<input_t> params{
          static_cast<const input_t*>(input.data_ptr()),
          output_q.data_ptr<int8_t>(),
          output_s.data_ptr(),
          CAL_SUM_V ? static_cast<input_t*>(output_sum->data_ptr()) : nullptr,
          hidden_dim,
          num_tokens,
          scale_dtype,
      };

      const bool can_use_cached =
          (vec_size == CACHED_VEC_SIZE) && (hidden_dim <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()));
      if (can_use_cached && try_launch_cached_kernel<input_t, CAL_SUM_V>(Q, params, xe_cores, max_wg)) {
        return;
      }

      // Fallback streaming 2-pass block kernel (unaligned, non-4-divisible, or huge hidden_dim).
      SGL_DISPATCH_VEC_SIZE_SET(vec_size, (1, 2, 4), [&] {
        sycl::range<1> global_range(static_cast<size_t>(num_tokens) * STREAM_BLOCK_WG_SIZE);
        sycl::range<1> local_range(STREAM_BLOCK_WG_SIZE);
        auto kernel = PerTokenQuantInt8StreamBlockKernel<input_t, STREAM_BLOCK_WG_SIZE, VEC_SIZE, CAL_SUM_V>(params);
        sycl_kernel_submit(global_range, local_range, Q, kernel);
      });
    });
  });
}
