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

static constexpr int SUB_GROUP_SIZE = 32;
static constexpr int TOKENS_PER_WG = 8;  // sub-groups (tokens) per work-group in the warp kernel
static constexpr int STREAM_BLOCK_WG_SIZE = 512;
constexpr float INT8_QUANT_MAX = 127.0f;
constexpr float ABSMAX_EPS = 1e-10f;

enum class ScaleDtype : int {
  FP32 = 0,
  FP16 = 1,
  BF16 = 2,
};

static inline ScaleDtype to_scale_dtype(at::ScalarType st) {
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

static inline void write_scale(void* output_s, int64_t token_id, float scale, ScaleDtype dtype) {
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

// Coalesced 4-element vector helper: emits a single LSC vector load (d32x2 for
// fp16/bf16, d32x4 for fp32) and a single 32-bit int8x4 store (d32) per lane.
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
// Single-pass register-cached warp kernel: one sub-group (32 lanes) handles
// one token with zero barriers and zero SLM when hidden_dim is small (<=1024)
// and num_tokens is large enough to saturate all Xe-cores.
// ---------------------------------------------------------------------------
template <typename T, int ITERS, bool CAL_SUM>
class PerTokenQuantInt8CachedWarpKernel {
  static constexpr int VEC_SIZE = 4;

 private:
  const T* input_;
  int8_t* output_q_;
  void* output_s_;
  T* output_sum_;
  uint32_t hidden_dim_;
  int64_t num_tokens_;
  ScaleDtype scale_dtype_;

 public:
  PerTokenQuantInt8CachedWarpKernel(
      const T* input,
      int8_t* output_q,
      void* output_s,
      T* output_sum,
      uint32_t hidden_dim,
      int64_t num_tokens,
      ScaleDtype scale_dtype)
      : input_(input),
        output_q_(output_q),
        output_s_(output_s),
        output_sum_(output_sum),
        hidden_dim_(hidden_dim),
        num_tokens_(num_tokens),
        scale_dtype_(scale_dtype) {}

  [[sycl::reqd_work_group_size(TOKENS_PER_WG * SUB_GROUP_SIZE)]] [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] void
  operator()(sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    const uint32_t tid = static_cast<uint32_t>(item.get_local_id(0));
    const uint32_t warp_id = tid / SUB_GROUP_SIZE;
    const uint32_t lane_id = tid % SUB_GROUP_SIZE;
    const int64_t token_id = static_cast<int64_t>(item.get_group(0)) * TOKENS_PER_WG + warp_id;
    if (token_id >= num_tokens_) return;

    const T* token_input = input_ + token_id * static_cast<int64_t>(hidden_dim_);
    int8_t* token_output = output_q_ + token_id * static_cast<int64_t>(hidden_dim_);
    const uint32_t num_vecs = hidden_dim_ / VEC_SIZE;

    using Pack = RawPack4<T>;
    using storage_t = typename Pack::storage_t;

    storage_t raw_reg[ITERS]{};
    if constexpr (ITERS > 1) {
#pragma unroll
      for (int it = 0; it < ITERS - 1; ++it) {
        const uint32_t vec_idx = static_cast<uint32_t>(it) * SUB_GROUP_SIZE + lane_id;
        raw_reg[it] = Pack::load(token_input + vec_idx * VEC_SIZE);
      }
    }
    const uint32_t last_vec_idx = static_cast<uint32_t>(ITERS - 1) * SUB_GROUP_SIZE + lane_id;
    if (last_vec_idx < num_vecs) {
      raw_reg[ITERS - 1] = Pack::load(token_input + last_vec_idx * VEC_SIZE);
    }

    float max_value = 0.0f;
    float sum_value = 0.0f;
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const auto elems = Pack::unpack(raw_reg[it]);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const float val = static_cast<float>(elems.v[j]);
        max_value = sycl::fmax(max_value, sycl::fabs(val));
        if constexpr (CAL_SUM) {
          sum_value += val;
        }
      }
    }

    max_value = sycl::fmax(sycl::reduce_over_group(sg, max_value, sycl::maximum<float>()), ABSMAX_EPS);
    if constexpr (CAL_SUM) {
      sum_value = sycl::reduce_over_group(sg, sum_value, sycl::plus<float>());
      if (lane_id == 0) {
        output_sum_[token_id] = static_cast<T>(sum_value);
      }
    }

    if (lane_id == 0) {
      write_scale(output_s_, token_id, max_value / INT8_QUANT_MAX, scale_dtype_);
    }
    const float scale_inv = INT8_QUANT_MAX / max_value;

    uint32_t q_packed[ITERS];
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      q_packed[it] = Pack::quantize_pack(raw_reg[it], scale_inv);
    }

    if constexpr (ITERS > 1) {
#pragma unroll
      for (int it = 0; it < ITERS - 1; ++it) {
        const uint32_t vec_idx = static_cast<uint32_t>(it) * SUB_GROUP_SIZE + lane_id;
        Pack::store(token_output + vec_idx * VEC_SIZE, q_packed[it]);
      }
    }
    if (last_vec_idx < num_vecs) {
      Pack::store(token_output + last_vec_idx * VEC_SIZE, q_packed[ITERS - 1]);
    }
  }
};

// ---------------------------------------------------------------------------
// Single-pass register-cached block kernel: one work-group (WG_SIZE lanes)
// handles one token. Keeps the row in GRF registers across a single-barrier
// SLM reduction so global memory is read only once — eliminating LLC thrashing
// in Long Prefill as well as Pass-2 L2 reloads.
// ---------------------------------------------------------------------------
template <typename T, int WG_SIZE, int ITERS, bool CAL_SUM>
class PerTokenQuantInt8CachedBlockKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int VEC_SIZE = 4;
  static constexpr int NUM_SGS = WG_SIZE / SUB_GROUP_SIZE;
  static_assert(WG_SIZE % SUB_GROUP_SIZE == 0 && NUM_SGS <= SUB_GROUP_SIZE);

 private:
  const T* input_;
  int8_t* output_q_;
  void* output_s_;
  T* output_sum_;
  uint32_t hidden_dim_;
  int64_t num_tokens_;
  ScaleDtype scale_dtype_;
  sycl::local_accessor<float, 1> slm_{};

 public:
  PerTokenQuantInt8CachedBlockKernel(
      const T* input,
      int8_t* output_q,
      void* output_s,
      T* output_sum,
      uint32_t hidden_dim,
      int64_t num_tokens,
      ScaleDtype scale_dtype)
      : input_(input),
        output_q_(output_q),
        output_s_(output_s),
        output_sum_(output_sum),
        hidden_dim_(hidden_dim),
        num_tokens_(num_tokens),
        scale_dtype_(scale_dtype) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    constexpr int slm_floats = (CAL_SUM ? 2 : 1) * NUM_SGS;
    slm_ = sycl::local_accessor<float, 1>(slm_floats, cgh);
  }

  [[sycl::reqd_work_group_size(WG_SIZE)]] [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] void
  operator()(sycl::nd_item<1> item) const {
    const int64_t token_id = static_cast<int64_t>(item.get_group(0));
    if (token_id >= num_tokens_) return;

    auto sg = item.get_sub_group();
    const uint32_t tid = static_cast<uint32_t>(item.get_local_id(0));
    const uint32_t sg_id = tid / SUB_GROUP_SIZE;
    const uint32_t lane_id = tid % SUB_GROUP_SIZE;

    const T* token_input = input_ + token_id * static_cast<int64_t>(hidden_dim_);
    int8_t* token_output = output_q_ + token_id * static_cast<int64_t>(hidden_dim_);
    const uint32_t num_vecs = hidden_dim_ / VEC_SIZE;

    using Pack = RawPack4<T>;
    using storage_t = typename Pack::storage_t;

    storage_t raw_reg[ITERS]{};
    if constexpr (ITERS > 1) {
#pragma unroll
      for (int it = 0; it < ITERS - 1; ++it) {
        const uint32_t vec_idx = static_cast<uint32_t>(it) * WG_SIZE + tid;
        raw_reg[it] = Pack::load(token_input + vec_idx * VEC_SIZE);
      }
    }
    const uint32_t last_vec_idx = static_cast<uint32_t>(ITERS - 1) * WG_SIZE + tid;
    if (last_vec_idx < num_vecs) {
      raw_reg[ITERS - 1] = Pack::load(token_input + last_vec_idx * VEC_SIZE);
    }

    float max_value = 0.0f;
    float sum_value = 0.0f;
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      const auto elems = Pack::unpack(raw_reg[it]);
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) {
        const float val = static_cast<float>(elems.v[j]);
        max_value = sycl::fmax(max_value, sycl::fabs(val));
        if constexpr (CAL_SUM) {
          sum_value += val;
        }
      }
    }

    max_value = sycl::reduce_over_group(sg, max_value, sycl::maximum<float>());
    if constexpr (CAL_SUM) {
      sum_value = sycl::reduce_over_group(sg, sum_value, sycl::plus<float>());
    }

    float* slm = slm_.template get_multi_ptr<sycl::access::decorated::no>().get();
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
          output_sum_[token_id] = static_cast<T>(sum_value);
        }
      }
    }

    if (tid == 0) {
      write_scale(output_s_, token_id, max_value / INT8_QUANT_MAX, scale_dtype_);
    }
    const float scale_inv = INT8_QUANT_MAX / max_value;

    uint32_t q_packed[ITERS];
#pragma unroll
    for (int it = 0; it < ITERS; ++it) {
      q_packed[it] = Pack::quantize_pack(raw_reg[it], scale_inv);
    }

    if constexpr (ITERS > 1) {
#pragma unroll
      for (int it = 0; it < ITERS - 1; ++it) {
        const uint32_t vec_idx = static_cast<uint32_t>(it) * WG_SIZE + tid;
        Pack::store(token_output + vec_idx * VEC_SIZE, q_packed[it]);
      }
    }
    if (last_vec_idx < num_vecs) {
      Pack::store(token_output + last_vec_idx * VEC_SIZE, q_packed[ITERS - 1]);
    }
  }
};

// ---------------------------------------------------------------------------
// Two-pass fallback block kernel for unaligned pointers, non-4-divisible
// hidden_dim, or extreme hidden_dim (>32768). Uses one work-group per token
// so concurrent active rows still fit comfortably inside LLC.
// ---------------------------------------------------------------------------
template <typename T, int WG_SIZE, int VEC_SIZE, bool CAL_SUM>
class PerTokenQuantInt8StreamBlockKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int NUM_SGS = WG_SIZE / SUB_GROUP_SIZE;
  static_assert(WG_SIZE % SUB_GROUP_SIZE == 0 && NUM_SGS <= SUB_GROUP_SIZE);

 private:
  const T* input_;
  int8_t* output_q_;
  void* output_s_;
  T* output_sum_;
  int64_t hidden_dim_;
  int64_t num_tokens_;
  ScaleDtype scale_dtype_;
  sycl::local_accessor<float, 1> slm_{};

 public:
  PerTokenQuantInt8StreamBlockKernel(
      const T* input,
      int8_t* output_q,
      void* output_s,
      T* output_sum,
      int64_t hidden_dim,
      int64_t num_tokens,
      ScaleDtype scale_dtype)
      : input_(input),
        output_q_(output_q),
        output_s_(output_s),
        output_sum_(output_sum),
        hidden_dim_(hidden_dim),
        num_tokens_(num_tokens),
        scale_dtype_(scale_dtype) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    constexpr int slm_floats = (CAL_SUM ? 2 : 1) * NUM_SGS;
    slm_ = sycl::local_accessor<float, 1>(slm_floats, cgh);
  }

  [[sycl::reqd_work_group_size(WG_SIZE)]] [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] void
  operator()(sycl::nd_item<1> item) const {
    const int64_t token_id = static_cast<int64_t>(item.get_group(0));
    if (token_id >= num_tokens_) return;

    auto sg = item.get_sub_group();
    const uint32_t tid = static_cast<uint32_t>(item.get_local_id(0));
    const uint32_t sg_id = tid / SUB_GROUP_SIZE;
    const uint32_t lane_id = tid % SUB_GROUP_SIZE;

    const T* token_input = input_ + token_id * hidden_dim_;
    int8_t* token_output = output_q_ + token_id * hidden_dim_;
    const int64_t num_vec_elems = hidden_dim_ / VEC_SIZE;

    using vec_in_t = vec_t<T, VEC_SIZE>;
    using vec_out_t = vec_t<int8_t, VEC_SIZE>;

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

    max_value = sycl::reduce_over_group(sg, max_value, sycl::maximum<float>());
    if constexpr (CAL_SUM) {
      sum_value = sycl::reduce_over_group(sg, sum_value, sycl::plus<float>());
    }

    float* slm = slm_.template get_multi_ptr<sycl::access::decorated::no>().get();
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
          output_sum_[token_id] = static_cast<T>(sum_value);
        }
      }
    }

    if (tid == 0) {
      write_scale(output_s_, token_id, max_value / INT8_QUANT_MAX, scale_dtype_);
    }
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

template <typename T, int WG_SIZE, bool CAL_SUM>
static void dispatch_cached_block_iters(
    int iters,
    sycl::queue& Q,
    const T* input_ptr,
    int8_t* output_q_ptr,
    void* output_s_ptr,
    T* sum_ptr,
    uint32_t hidden_dim,
    int64_t num_tokens,
    ScaleDtype scale_dtype) {
  auto launch = [&](auto iters_tag) {
    constexpr int ITERS = decltype(iters_tag)::value;
    sycl::range<1> global_range(static_cast<size_t>(num_tokens) * WG_SIZE);
    sycl::range<1> local_range(WG_SIZE);
    auto kernel = PerTokenQuantInt8CachedBlockKernel<T, WG_SIZE, ITERS, CAL_SUM>(
        input_ptr, output_q_ptr, output_s_ptr, sum_ptr, hidden_dim, num_tokens, scale_dtype);
    sycl_kernel_submit(global_range, local_range, Q, kernel);
  };

  if (iters <= 1) {
    launch(std::integral_constant<int, 1>{});
  } else if (iters == 2) {
    launch(std::integral_constant<int, 2>{});
  } else if (iters == 3) {
    launch(std::integral_constant<int, 3>{});
  } else if (iters == 4) {
    launch(std::integral_constant<int, 4>{});
  } else if constexpr (sizeof(T) == 2) {
    if (iters == 5) {
      launch(std::integral_constant<int, 5>{});
    } else if (iters == 6) {
      launch(std::integral_constant<int, 6>{});
    } else if (iters == 7) {
      launch(std::integral_constant<int, 7>{});
    } else {
      launch(std::integral_constant<int, 8>{});
    }
  } else {
    TORCH_CHECK(false, "Internal error: invalid cached block iters=", iters);
  }
}

template <typename T, bool CAL_SUM>
static void dispatch_cached_warp_iters(
    int iters,
    sycl::queue& Q,
    const T* input_ptr,
    int8_t* output_q_ptr,
    void* output_s_ptr,
    T* sum_ptr,
    uint32_t hidden_dim,
    int64_t num_tokens,
    ScaleDtype scale_dtype) {
  auto launch = [&](auto iters_tag) {
    constexpr int ITERS = decltype(iters_tag)::value;
    constexpr int wg_size = TOKENS_PER_WG * SUB_GROUP_SIZE;
    const int64_t num_wgs = (num_tokens + TOKENS_PER_WG - 1) / TOKENS_PER_WG;
    sycl::range<1> global_range(static_cast<size_t>(num_wgs) * wg_size);
    sycl::range<1> local_range(wg_size);
    auto kernel = PerTokenQuantInt8CachedWarpKernel<T, ITERS, CAL_SUM>(
        input_ptr, output_q_ptr, output_s_ptr, sum_ptr, hidden_dim, num_tokens, scale_dtype);
    sycl_kernel_submit(global_range, local_range, Q, kernel);
  };

  if (iters <= 1) {
    launch(std::integral_constant<int, 1>{});
  } else if (iters == 2) {
    launch(std::integral_constant<int, 2>{});
  } else if (iters == 3) {
    launch(std::integral_constant<int, 3>{});
  } else if (iters == 4) {
    launch(std::integral_constant<int, 4>{});
  } else if constexpr (sizeof(T) == 2) {
    if (iters == 5) {
      launch(std::integral_constant<int, 5>{});
    } else if (iters == 6) {
      launch(std::integral_constant<int, 6>{});
    } else if (iters == 7) {
      launch(std::integral_constant<int, 7>{});
    } else {
      launch(std::integral_constant<int, 8>{});
    }
  } else {
    TORCH_CHECK(false, "Internal error: invalid cached warp iters=", iters);
  }
}

SGL_KERNEL_EXPORT void sgl_per_token_quant_int8(
    at::Tensor input, at::Tensor output_q, at::Tensor output_s, const std::optional<at::Tensor>& output_sum) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_q);
  CHECK_INPUT(output_s);
  TORCH_CHECK(input.dim() >= 1 && input.size(-1) > 0, "input must have a nonempty last dimension");
  TORCH_CHECK(
      input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kFloat,
      "input must be Float16, BFloat16, or Float32");
  TORCH_CHECK(output_q.device() == input.device(), "output_q must be on the same device as input");
  TORCH_CHECK(output_s.device() == input.device(), "output_s must be on the same device as input");
  TORCH_CHECK(output_q.scalar_type() == at::kChar, "output_q must be Int8");
  TORCH_CHECK(output_q.sizes() == input.sizes(), "output_q must have the same shape as input");
  TORCH_CHECK(
      output_s.scalar_type() == at::kHalf || output_s.scalar_type() == at::kBFloat16 ||
          output_s.scalar_type() == at::kFloat,
      "output_s must be Float16, BFloat16, or Float32");
  auto scale_shape = input.sizes().vec();
  scale_shape.back() = 1;
  TORCH_CHECK(output_s.sizes() == at::IntArrayRef(scale_shape), "output_s must have shape input.shape[:-1] + (1,)");
  at::assert_no_overlap(input, output_q);
  at::assert_no_overlap(input, output_s);
  at::assert_no_overlap(output_q, output_s);

  const bool cal_sum = output_sum.has_value();
  if (cal_sum) {
    const auto& sums = *output_sum;
    CHECK_INPUT(sums);
    TORCH_CHECK(sums.device() == input.device(), "output_sum must be on the same device as input");
    TORCH_CHECK(sums.scalar_type() == input.scalar_type(), "output_sum must have the same dtype as input");
    auto sum_shape = input.sizes().vec();
    sum_shape.pop_back();
    TORCH_CHECK(sums.sizes() == at::IntArrayRef(sum_shape), "output_sum must have shape input.shape[:-1]");
    at::assert_no_overlap(input, sums);
    at::assert_no_overlap(output_q, sums);
    at::assert_no_overlap(output_s, sums);
  }

  const int64_t hidden_dim = input.size(-1);
  const int64_t num_tokens = input.numel() / hidden_dim;
  if (num_tokens == 0) return;

  const c10::OptionalDeviceGuard guard(input.device());
  auto& Q = dpcppGetCurrentQueue();
  const DeviceId dev_id = dpcppGetDeviceIdOfCurrentQueue();
  const ScaleDtype scale_dtype = to_scale_dtype(output_s.scalar_type());

  const uintptr_t in_addr = reinterpret_cast<uintptr_t>(input.data_ptr());
  const uintptr_t out_addr = reinterpret_cast<uintptr_t>(output_q.data_ptr());
  const size_t elem_size = static_cast<size_t>(input.element_size());

  int vec_size = 4;
  while (vec_size > 1 && ((hidden_dim % vec_size != 0) || (in_addr % (vec_size * elem_size) != 0) ||
                          (out_addr % static_cast<size_t>(vec_size) != 0))) {
    vec_size /= 2;
  }

  const int64_t xe_cores = dpcppMaxComputeUnitSize(dev_id);
  const int64_t max_wg = dpcppMaxWorkGroupSize(dev_id);

#if defined(CUTLASS_SYCL_PROFILING_ENABLED)
  // Per-token INT8 quant: one scale/mul (+ optional add) per input element. Memory-bound.
  const double flops = static_cast<double>(input.numel()) * (cal_sum ? 2.0 : 1.0);
  double bytes = static_cast<double>(input.numel()) * static_cast<double>(input.element_size()) +
                 static_cast<double>(output_q.numel()) * static_cast<double>(output_q.element_size()) +
                 static_cast<double>(output_s.numel()) * static_cast<double>(output_s.element_size());
  if (cal_sum) {
    bytes += static_cast<double>(output_sum->numel()) * static_cast<double>(output_sum->element_size());
  }
  auto profiling_queue = at::xpu::getCurrentXPUStream().queue();
  SGL_KERNEL_PERF_SCOPE("per_token_quant_int8", profiling_queue, bytes, flops);
#endif

  AT_DISPATCH_FLOATING_TYPES_V2(input.scalar_type(), "sgl_per_token_quant_int8_input", [&] {
    using input_t = sycl_type_t<scalar_t>;
    SGL_DISPATCH_BOOL(cal_sum, CAL_SUM_V, [&] {
      const input_t* input_ptr = static_cast<const input_t*>(input.data_ptr());
      int8_t* output_q_ptr = output_q.data_ptr<int8_t>();
      void* output_s_ptr = output_s.data_ptr();
      input_t* sum_ptr = CAL_SUM_V ? static_cast<input_t*>(output_sum->data_ptr()) : nullptr;

      constexpr int max_warp_iters = (sizeof(input_t) == 2) ? 8 : 4;
      constexpr int max_iters_narrow = (sizeof(input_t) == 2) ? 7 : 4;
      constexpr int max_iters_wide = (sizeof(input_t) == 2) ? 8 : 4;

      if (vec_size == 4 && hidden_dim <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        const uint32_t u_hidden = static_cast<uint32_t>(hidden_dim);
        const int64_t num_vecs = hidden_dim / 4;

        // Small rows at large batch: 1 sub-group per token, register-cached, 0 barriers.
        if (num_tokens >= xe_cores * 2 * TOKENS_PER_WG && num_vecs <= SUB_GROUP_SIZE * max_warp_iters) {
          const int iters = static_cast<int>((num_vecs + SUB_GROUP_SIZE - 1) / SUB_GROUP_SIZE);
          dispatch_cached_warp_iters<input_t, CAL_SUM_V>(
              iters, Q, input_ptr, output_q_ptr, output_s_ptr, sum_ptr, u_hidden, num_tokens, scale_dtype);
          return;
        }

        // General single-pass register-cached block path: 1 work-group per token.
        if (num_vecs <= 128 * max_iters_narrow && !(num_tokens < xe_cores && num_vecs > 128)) {
          const int iters = static_cast<int>((num_vecs + 127) / 128);
          dispatch_cached_block_iters<input_t, 128, CAL_SUM_V>(
              iters, Q, input_ptr, output_q_ptr, output_s_ptr, sum_ptr, u_hidden, num_tokens, scale_dtype);
          return;
        }
        if (num_vecs <= 256 * max_iters_narrow && !(num_tokens <= 32 && num_vecs > 512)) {
          const int iters = static_cast<int>((num_vecs + 255) / 256);
          dispatch_cached_block_iters<input_t, 256, CAL_SUM_V>(
              iters, Q, input_ptr, output_q_ptr, output_s_ptr, sum_ptr, u_hidden, num_tokens, scale_dtype);
          return;
        }
        if (num_vecs <= 512 * max_iters_wide) {
          const int iters = static_cast<int>((num_vecs + 511) / 512);
          dispatch_cached_block_iters<input_t, 512, CAL_SUM_V>(
              iters, Q, input_ptr, output_q_ptr, output_s_ptr, sum_ptr, u_hidden, num_tokens, scale_dtype);
          return;
        }
        if constexpr (sizeof(input_t) == 2) {
          if (max_wg >= 1024 && num_vecs <= 1024 * max_iters_wide) {
            const int iters = static_cast<int>((num_vecs + 1023) / 1024);
            dispatch_cached_block_iters<input_t, 1024, CAL_SUM_V>(
                iters, Q, input_ptr, output_q_ptr, output_s_ptr, sum_ptr, u_hidden, num_tokens, scale_dtype);
            return;
          }
        }
      }

      // Fallback streaming 2-pass block kernel (unaligned, non-4-divisible, or huge hidden_dim).
      SGL_DISPATCH_VEC_SIZE_SET(vec_size, (1, 2, 4), [&] {
        sycl::range<1> global_range(static_cast<size_t>(num_tokens) * STREAM_BLOCK_WG_SIZE);
        sycl::range<1> local_range(STREAM_BLOCK_WG_SIZE);
        auto kernel = PerTokenQuantInt8StreamBlockKernel<input_t, STREAM_BLOCK_WG_SIZE, VEC_SIZE, CAL_SUM_V>(
            input_ptr, output_q_ptr, output_s_ptr, sum_ptr, hidden_dim, num_tokens, scale_dtype);
        sycl_kernel_submit(global_range, local_range, Q, kernel);
      });
    });
  });
}
