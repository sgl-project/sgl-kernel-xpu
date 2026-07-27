#include <ATen/ATen.h>
#include <torch/all.h>

#include <limits>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

static constexpr int WG_SIZE = 256;
static constexpr int HC_MULT = 4;
static constexpr float LOG2E = 1.442695040888963f;

template <typename scalar_t>
struct FusedHCHeadKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  const scalar_t* __restrict__ x;  // [T, HC, D]
  const float* __restrict__ hc_fn;  // [HC, HC*D]
  const float* __restrict__ hc_scale;  // [1]
  const float* __restrict__ hc_base;  // [HC]
  scalar_t* __restrict__ y;  // [T, D]

  int T;
  int hidden_size;
  int k_total;
  float norm_eps;
  float hc_eps;

  sycl::local_accessor<float, 1> slm_;

  FusedHCHeadKernel(
      const scalar_t* x_,
      const float* hc_fn_,
      const float* hc_scale_,
      const float* hc_base_,
      scalar_t* y_,
      int T_,
      int hidden_size_,
      float norm_eps_,
      float hc_eps_)
      : x(x_),
        hc_fn(hc_fn_),
        hc_scale(hc_scale_),
        hc_base(hc_base_),
        y(y_),
        T(T_),
        hidden_size(hidden_size_),
        k_total(HC_MULT * hidden_size_),
        norm_eps(norm_eps_),
        hc_eps(hc_eps_) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    constexpr int slm_size = (HC_MULT + 1) * WG_SIZE + HC_MULT;
    slm_ = sycl::local_accessor<float, 1>(slm_size, cgh);
  }

  [[sycl::reqd_work_group_size(WG_SIZE)]] void operator()(sycl::nd_item<1> item) const {
    const int token_id = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    if (token_id >= T) {
      return;
    }

    float* slm = slm_.template get_multi_ptr<sycl::access::decorated::no>().get();
    float* sumsq_buf = slm;
    float* mix_buf = slm + WG_SIZE;
    float* pre_buf = slm + (HC_MULT + 1) * WG_SIZE;

    float local_sumsq = 0.0f;
    float local_mix[HC_MULT] = {0.0f, 0.0f, 0.0f, 0.0f};

    const int64_t x_row_off = static_cast<int64_t>(token_id) * k_total;
    const scalar_t* x_row = x + x_row_off;

    for (int k = tid; k < k_total; k += WG_SIZE) {
      const float xv = static_cast<float>(x_row[k]);
      local_sumsq += xv * xv;

#pragma unroll
      for (int m = 0; m < HC_MULT; ++m) {
        local_mix[m] += hc_fn[static_cast<int64_t>(m) * k_total + k] * xv;
      }
    }

    sumsq_buf[tid] = local_sumsq;
#pragma unroll
    for (int m = 0; m < HC_MULT; ++m) {
      mix_buf[m * WG_SIZE + tid] = local_mix[m];
    }

    item.barrier(sycl::access::fence_space::local_space);

    for (int stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        sumsq_buf[tid] += sumsq_buf[tid + stride];
#pragma unroll
        for (int m = 0; m < HC_MULT; ++m) {
          mix_buf[m * WG_SIZE + tid] += mix_buf[m * WG_SIZE + tid + stride];
        }
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    if (tid == 0) {
      const float rsqrt = sycl::native::rsqrt(sumsq_buf[0] / static_cast<float>(k_total) + norm_eps);
      const float scale = hc_scale[0];
#pragma unroll
      for (int m = 0; m < HC_MULT; ++m) {
        const float logit = mix_buf[m * WG_SIZE] * rsqrt * scale + hc_base[m];
        pre_buf[m] = sycl::native::recip(1.0f + sycl::native::exp2(-logit * LOG2E)) + hc_eps;
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    scalar_t* y_row = y + static_cast<int64_t>(token_id) * hidden_size;
    for (int d = tid; d < hidden_size; d += WG_SIZE) {
      float acc = 0.0f;
#pragma unroll
      for (int m = 0; m < HC_MULT; ++m) {
        const float xv = static_cast<float>(x_row[m * hidden_size + d]);
        acc += pre_buf[m] * xv;
      }
      y_row[d] = static_cast<scalar_t>(acc);
    }
  }
};

template <typename scalar_t>
static at::Tensor launch_fused_hc_head(
    sycl::queue& q,
    const at::Tensor& x,
    const at::Tensor& hc_fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    int64_t T,
    int64_t D,
    float norm_eps,
    float hc_eps) {
  auto y = at::empty({T, D}, x.options());
  if (T == 0) {
    return y;
  }

  auto ker = FusedHCHeadKernel<scalar_t>(
      reinterpret_cast<const scalar_t*>(x.data_ptr()),
      hc_fn.data_ptr<float>(),
      hc_scale.data_ptr<float>(),
      hc_base.data_ptr<float>(),
      reinterpret_cast<scalar_t*>(y.data_ptr()),
      static_cast<int>(T),
      static_cast<int>(D),
      norm_eps,
      hc_eps);

  sycl_kernel_submit(T * WG_SIZE, static_cast<int64_t>(WG_SIZE), q, ker);
  return y;
}

at::Tensor fused_hc_head(
    const at::Tensor& x,
    const at::Tensor& hc_fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    double norm_eps,
    double hc_eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(hc_fn);
  CHECK_INPUT(hc_scale);
  CHECK_INPUT(hc_base);

  TORCH_CHECK(x.dim() == 3, "x must be [T, hc_mult, hidden_size], got shape=", x.sizes());
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf, "x must be bf16/fp16");
  TORCH_CHECK(hc_fn.scalar_type() == at::kFloat, "hc_fn must be float32");
  TORCH_CHECK(hc_scale.scalar_type() == at::kFloat, "hc_scale must be float32");
  TORCH_CHECK(hc_base.scalar_type() == at::kFloat, "hc_base must be float32");

  const int64_t T = x.size(0);
  const int64_t hc_mult = x.size(1);
  const int64_t D = x.size(2);
  TORCH_CHECK(hc_mult == HC_MULT, "fused_hc_head currently supports hc_mult=", HC_MULT, ", got ", hc_mult);

  TORCH_CHECK(hc_fn.dim() == 2, "hc_fn must be 2D [hc_mult, hc_mult*hidden_size]");
  TORCH_CHECK(
      hc_fn.size(0) == hc_mult && hc_fn.size(1) == hc_mult * D,
      "hc_fn shape mismatch, expected [",
      hc_mult,
      ", ",
      hc_mult * D,
      "], got ",
      hc_fn.sizes());
  TORCH_CHECK(hc_scale.numel() == 1, "hc_scale must have one element");
  TORCH_CHECK(hc_base.numel() == hc_mult, "hc_base must have hc_mult elements");

  TORCH_CHECK(T < std::numeric_limits<int>::max(), "T must fit in int32, got ", T);
  TORCH_CHECK(D < std::numeric_limits<int>::max(), "hidden_size must fit in int32, got ", D);

  auto q = dpcppGetCurrentQueue();
  if (x.scalar_type() == at::kBFloat16) {
    return launch_fused_hc_head<sycl::ext::oneapi::bfloat16>(
        q, x, hc_fn, hc_scale, hc_base, T, D, static_cast<float>(norm_eps), static_cast<float>(hc_eps));
  }

  return launch_fused_hc_head<sycl::half>(
      q, x, hc_fn, hc_scale, hc_base, T, D, static_cast<float>(norm_eps), static_cast<float>(hc_eps));
}
