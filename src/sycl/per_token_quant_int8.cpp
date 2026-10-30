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

#include <optional>
#include <sycl/sycl.hpp>

#include "MemoryAccess.h"
#include "SGLKernelPerf.h"
#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_export.h"

static constexpr int SUB_GROUP_SIZE = 32;
static constexpr int TOKENS_PER_WG = 8;  // sub-groups (tokens) per work-group in the warp kernel
static constexpr int BLOCK_WG_SIZE = 256;
constexpr float INT8_QUANT_MAX = 127.0f;
constexpr float ABSMAX_EPS = 1e-10f;

// ---------------------------------------------------------------------------
// Warp kernel: one sub-group (32 lanes) handles one token.
// TOKENS_PER_WG sub-groups per work-group. Two passes over global memory
// (absmax [+ optional sum], then quantize); pass-2 reload hits L2 for typical
// hidden sizes.
// ---------------------------------------------------------------------------
template <typename T, typename Scale, int VEC_SIZE, bool CAL_SUM>
class PerTokenQuantInt8WarpKernel {
 private:
  const T* input_;
  int8_t* output_q_;
  Scale* output_s_;
  T* output_sum_;
  int64_t hidden_dim_;
  int64_t num_tokens_;

 public:
  PerTokenQuantInt8WarpKernel(
      const T* input, int8_t* output_q, Scale* output_s, T* output_sum, int64_t hidden_dim, int64_t num_tokens)
      : input_(input),
        output_q_(output_q),
        output_s_(output_s),
        output_sum_(output_sum),
        hidden_dim_(hidden_dim),
        num_tokens_(num_tokens) {}

  [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] void operator()(sycl::nd_item<1> item) const {
    auto sg = item.get_sub_group();
    const int warp_id = item.get_local_id(0) / SUB_GROUP_SIZE;
    const int lane_id = item.get_local_id(0) % SUB_GROUP_SIZE;
    const int64_t token_id = static_cast<int64_t>(item.get_group(0)) * TOKENS_PER_WG + warp_id;
    if (token_id >= num_tokens_) return;

    const T* token_input = input_ + token_id * hidden_dim_;
    int8_t* token_output = output_q_ + token_id * hidden_dim_;
    const int64_t num_vec_elems = hidden_dim_ / VEC_SIZE;

    using vec_in_t = vec_t<T, VEC_SIZE>;
    using vec_out_t = vec_t<int8_t, VEC_SIZE>;

    // Pass 1: strided vectorized load, per-lane absmax (and optional sum).
    float max_value = 0.0f;
    float sum_value = 0.0f;
    for (int64_t i = lane_id; i < num_vec_elems; i += SUB_GROUP_SIZE) {
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

    // Reduce across the sub-group (one token).
    max_value = sycl::fmax(sycl::reduce_over_group(sg, max_value, sycl::maximum<float>()), ABSMAX_EPS);
    if constexpr (CAL_SUM) {
      sum_value = sycl::reduce_over_group(sg, sum_value, sycl::plus<float>());
      if (lane_id == 0) {
        output_sum_[token_id] = static_cast<T>(sum_value);
      }
    }

    if (lane_id == 0) {
      output_s_[token_id] = static_cast<Scale>(max_value / INT8_QUANT_MAX);
    }
    const float scale_inv = INT8_QUANT_MAX / max_value;

    // Pass 2: reload, quantize, vectorized store.
    for (int64_t i = lane_id; i < num_vec_elems; i += SUB_GROUP_SIZE) {
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

// ---------------------------------------------------------------------------
// Small-batch kernel: one work-group handles one token, block reduce.
// Used when there are too few tokens to fill the machine with the warp kernel.
// ---------------------------------------------------------------------------
template <typename T, typename Scale, int VEC_SIZE, bool CAL_SUM>
class PerTokenQuantInt8BlockKernel {
 private:
  const T* input_;
  int8_t* output_q_;
  Scale* output_s_;
  T* output_sum_;
  int64_t hidden_dim_;
  int64_t num_tokens_;

 public:
  PerTokenQuantInt8BlockKernel(
      const T* input, int8_t* output_q, Scale* output_s, T* output_sum, int64_t hidden_dim, int64_t num_tokens)
      : input_(input),
        output_q_(output_q),
        output_s_(output_s),
        output_sum_(output_sum),
        hidden_dim_(hidden_dim),
        num_tokens_(num_tokens) {}

  [[sycl::reqd_sub_group_size(SUB_GROUP_SIZE)]] void operator()(sycl::nd_item<1> item) const {
    const int64_t token_id = static_cast<int64_t>(item.get_group(0));
    if (token_id >= num_tokens_) return;

    const int tid = item.get_local_id(0);
    const int block_dim = item.get_local_range(0);

    const T* token_input = input_ + token_id * hidden_dim_;
    int8_t* token_output = output_q_ + token_id * hidden_dim_;
    const int64_t num_vec_elems = hidden_dim_ / VEC_SIZE;

    using vec_in_t = vec_t<T, VEC_SIZE>;
    using vec_out_t = vec_t<int8_t, VEC_SIZE>;

    float max_value = 0.0f;
    float sum_value = 0.0f;
    for (int64_t i = tid; i < num_vec_elems; i += block_dim) {
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

    max_value = sycl::fmax(sycl::reduce_over_group(item.get_group(), max_value, sycl::maximum<float>()), ABSMAX_EPS);
    if constexpr (CAL_SUM) {
      sum_value = sycl::reduce_over_group(item.get_group(), sum_value, sycl::plus<float>());
      if (tid == 0) {
        output_sum_[token_id] = static_cast<T>(sum_value);
      }
    }

    if (tid == 0) {
      output_s_[token_id] = static_cast<Scale>(max_value / INT8_QUANT_MAX);
    }
    const float scale_inv = INT8_QUANT_MAX / max_value;

    for (int64_t i = tid; i < num_vec_elems; i += block_dim) {
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
  const int dev_id = dpcppGetDeviceIdOfCurrentQueue();

  // Choose vector width from device preference, capped by hidden_dim divisibility and pointer alignment.
  int vec_size = preferred_vector_width(dev_id, input.element_size());
  const auto align_addr =
      reinterpret_cast<uintptr_t>(input.data_ptr()) | reinterpret_cast<uintptr_t>(output_q.data_ptr());
  while (vec_size > 1 && ((hidden_dim % vec_size != 0) ||
                          (align_addr % (static_cast<uintptr_t>(vec_size) * input.element_size()) != 0))) {
    vec_size /= 2;
  }

  // Enough tokens to fill the machine → warp kernel (one sub-group per token).
  // Otherwise the block kernel (one work-group per token) gives more parallelism
  // across the hidden dimension for the few tokens present.
  const int64_t xe_cores = dpcppMaxComputeUnitSize(dev_id);
  const bool use_warp_kernel = num_tokens >= xe_cores * 2 * TOKENS_PER_WG;

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

#define LAUNCH_WARP(T, SCALE_T, VEC, CAL_SUM_FLAG)                                 \
  do {                                                                             \
    const int wg_size = TOKENS_PER_WG * SUB_GROUP_SIZE;                            \
    const int64_t num_wgs = (num_tokens + TOKENS_PER_WG - 1) / TOKENS_PER_WG;      \
    sycl::range<1> global_range(num_wgs* wg_size);                                 \
    sycl::range<1> local_range(wg_size);                                           \
    T* sum_ptr = CAL_SUM_FLAG ? static_cast<T*>(output_sum->data_ptr()) : nullptr; \
    auto kernel = PerTokenQuantInt8WarpKernel<T, SCALE_T, VEC, CAL_SUM_FLAG>(      \
        static_cast<const T*>(input.data_ptr()),                                   \
        output_q.data_ptr<int8_t>(),                                               \
        static_cast<SCALE_T*>(output_s.data_ptr()),                                \
        sum_ptr,                                                                   \
        hidden_dim,                                                                \
        num_tokens);                                                               \
    sycl_kernel_submit(global_range, local_range, Q, kernel);                      \
  } while (0)

#define LAUNCH_BLOCK(T, SCALE_T, VEC, CAL_SUM_FLAG)                                \
  do {                                                                             \
    const int wg_size = BLOCK_WG_SIZE;                                             \
    sycl::range<1> global_range(num_tokens* wg_size);                              \
    sycl::range<1> local_range(wg_size);                                           \
    T* sum_ptr = CAL_SUM_FLAG ? static_cast<T*>(output_sum->data_ptr()) : nullptr; \
    auto kernel = PerTokenQuantInt8BlockKernel<T, SCALE_T, VEC, CAL_SUM_FLAG>(     \
        static_cast<const T*>(input.data_ptr()),                                   \
        output_q.data_ptr<int8_t>(),                                               \
        static_cast<SCALE_T*>(output_s.data_ptr()),                                \
        sum_ptr,                                                                   \
        hidden_dim,                                                                \
        num_tokens);                                                               \
    sycl_kernel_submit(global_range, local_range, Q, kernel);                      \
  } while (0)

  AT_DISPATCH_FLOATING_TYPES_V2(input.scalar_type(), "sgl_per_token_quant_int8_input", [&] {
    using input_t = sycl_type_t<scalar_t>;
    AT_DISPATCH_FLOATING_TYPES_V2(output_s.scalar_type(), "sgl_per_token_quant_int8_scale", [&] {
      using scale_t = sycl_type_t<scalar_t>;
      SGL_DISPATCH_VEC_SIZE(vec_size, [&] {
        SGL_DISPATCH_BOOL(cal_sum, CAL_SUM_V, [&] {
          if (use_warp_kernel) {
            LAUNCH_WARP(input_t, scale_t, VEC_SIZE, CAL_SUM_V);
          } else {
            LAUNCH_BLOCK(input_t, scale_t, VEC_SIZE, CAL_SUM_V);
          }
        });
      });
    });
  });

#undef LAUNCH_BLOCK
#undef LAUNCH_WARP
}
