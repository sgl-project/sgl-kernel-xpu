#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <ATen/Parallel.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

#include "MemoryAccess.h"
#include "SYCLHelpers.h"
#include "Utils.h"

#define DPCPP_CONSTANT __attribute__((opencl_constant))

#define DPCPP_KER_STRING(var, str) static const DPCPP_CONSTANT char var[] = str;
#define DPCPP_KER_PRINTF sycl::ext::oneapi::experimental::printf

#define DPCPP_K_PRINT(fmt_str, ...)           \
  {                                           \
    DPCPP_KER_STRING(fmt_var, fmt_str);       \
    DPCPP_KER_PRINTF(fmt_var, ##__VA_ARGS__); \
  }

template <typename scalar_t, typename accscalar_t>
struct silu_mul_dpcpp_functor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return (accscalar_t(a)) / (1.0f + expf(accscalar_t(-a))) * accscalar_t(b);
  }
};

template <typename scalar_t, typename accscalar_t>
struct gelu_tanh_mul_dpcpp_functor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    const accscalar_t kBeta = M_SQRT2 * M_2_SQRTPI * accscalar_t(0.5);
    const accscalar_t kKappa = 0.044715;
    auto x_cube = accscalar_t(a) * accscalar_t(a) * accscalar_t(a);
    auto inner = kBeta * (accscalar_t(a) + kKappa * x_cube);
    return (accscalar_t(0.5) * accscalar_t(a) * (accscalar_t(1) + std::tanh(accscalar_t(inner)))) * accscalar_t(b);
  }
};

template <typename scalar_t, typename accscalar_t>
struct gelu_erf_mul_dpcpp_functor {
  scalar_t operator()(scalar_t a, scalar_t b) const {
    return (accscalar_t(a) * accscalar_t(0.5) * (accscalar_t(1) + ::erf(accscalar_t(a) * accscalar_t(M_SQRT1_2)))) *
           accscalar_t(b);
  }
};

template <typename scalar_t, typename func_t, int N>
struct op_and_mul_functor {
  void operator()(sycl::nd_item<1> item) const {
    using accscalar_t = at::opmath_type<scalar_t>;
    int64_t offset = item.get_local_linear_id();
    int64_t step = item.get_local_range(0);
    int64_t token_id = item.get_group(0);
    func_t fn;
    int64_t bound = dim / N;
    for (int64_t i = offset; i < bound; i += step) {
      auto unary_val = reinterpret_cast<aligned_vector_loop<scalar_t, N>*>(input_ptr)[token_id * bound * 2 + i];
      auto mul_val = reinterpret_cast<aligned_vector_loop<scalar_t, N>*>(input_ptr)[token_id * bound * 2 + i + bound];
#pragma unroll
      for (int i = 0; i < N; ++i) {
        auto a = unary_val[i], b = mul_val[i];
        unary_val[i] = fn(unary_val[i], mul_val[i]);
      }
      reinterpret_cast<aligned_vector_loop<scalar_t, N>*>(output_ptr)[token_id * bound + i] = unary_val;
    }
  }

  scalar_t* input_ptr;
  scalar_t* output_ptr;
  int64_t num_;
  int64_t dim;
};

#define VEC_LAUNCH(KERNEL, N)                                                \
  case N: {                                                                  \
    op_and_mul_functor<T_to, KERNEL<T_to, accscalar_t>, N> kfn = {           \
        .input_ptr = _input, .output_ptr = _out, .num_ = numel, .dim = dim}; \
    sycl_kernel_submit(num_group* wg_size, wg_size, q, kfn);                 \
    break;                                                                   \
  }

template <typename T = float>
void get_config(
    const at::Tensor& input,
    const at::Tensor& out,
    int64_t& numel,
    int64_t& dim,
    int64_t& wg_size,
    int64_t& num_group,
    int& vec_size) {
  auto dev_id = dpcppGetDeviceIdOfCurrentQueue();
  int64_t max_wg_size = dpcppMaxWorkGroupSize(dev_id);
  numel = out.numel();
  dim = out.size(-1);
  int64_t tokens = numel / dim;
  wg_size = std::min(dim, max_wg_size);
  num_group = tokens;

  vec_size = sizeof(float) * 4 / sizeof(T);
  while ((vec_size >> 1) * wg_size >= dim) {
    vec_size = vec_size >> 1;
  }
  if (dim % vec_size != 0) vec_size = 1;
}

template <typename T_to = float, typename T_from = float>
void silu_and_mul_sycl(sycl::queue& q, at::Tensor& input, at::Tensor& out) {
  auto _input = reinterpret_cast<T_to*>(input.data_ptr<T_from>());
  auto _out = reinterpret_cast<T_to*>(out.data_ptr<T_from>());

  int64_t numel;
  int64_t dim;
  int64_t wg_size;
  int64_t num_group;
  int vec_size;
  get_config<T_to>(input, out, numel, dim, wg_size, num_group, vec_size);

  using accscalar_t = at::opmath_type<T_to>;
  switch (vec_size) {
    VEC_LAUNCH(silu_mul_dpcpp_functor, 1);
    VEC_LAUNCH(silu_mul_dpcpp_functor, 2);
    VEC_LAUNCH(silu_mul_dpcpp_functor, 4);
    VEC_LAUNCH(silu_mul_dpcpp_functor, 8);
    VEC_LAUNCH(silu_mul_dpcpp_functor, 16);
    default:
      TORCH_CHECK(false, "Unsupported vector size: ", vec_size);
  }

  return;
}

void silu_and_mul(at::Tensor& out, at::Tensor& input) {
  input = input.contiguous();
  out = out.contiguous();

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  if (input.scalar_type() == at::ScalarType::Half) {
    silu_and_mul_sycl<sycl::half, at::Half>(queue, input, out);
  } else {
    silu_and_mul_sycl<sycl::ext::oneapi::bfloat16, at::BFloat16>(queue, input, out);
  }
  return;
}

template <typename T_to = float, typename T_from = float>
void gelu_tanh_and_mul_sycl(sycl::queue& q, at::Tensor& input, at::Tensor& out) {
  auto _input = reinterpret_cast<T_to*>(input.data_ptr<T_from>());
  auto _out = reinterpret_cast<T_to*>(out.data_ptr<T_from>());

  int64_t numel;
  int64_t dim;
  int64_t wg_size;
  int64_t num_group;
  int vec_size;
  get_config<T_to>(input, out, numel, dim, wg_size, num_group, vec_size);

  using accscalar_t = at::opmath_type<T_to>;
  switch (vec_size) {
    VEC_LAUNCH(gelu_tanh_mul_dpcpp_functor, 1);
    VEC_LAUNCH(gelu_tanh_mul_dpcpp_functor, 2);
    VEC_LAUNCH(gelu_tanh_mul_dpcpp_functor, 4);
    VEC_LAUNCH(gelu_tanh_mul_dpcpp_functor, 8);
    VEC_LAUNCH(gelu_tanh_mul_dpcpp_functor, 16);
    default:
      TORCH_CHECK(false, "Unsupported vector size: ", vec_size);
  }

  return;
}

void gelu_tanh_and_mul(at::Tensor& out, at::Tensor& input) {
  input = input.contiguous();
  out = out.contiguous();

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  if (input.scalar_type() == at::ScalarType::Half) {
    gelu_tanh_and_mul_sycl<sycl::half, at::Half>(queue, input, out);
  } else {
    gelu_tanh_and_mul_sycl<sycl::ext::oneapi::bfloat16, at::BFloat16>(queue, input, out);
  }
  return;
}

template <typename T_to = float, typename T_from = float>
void gelu_and_mul_sycl(sycl::queue& q, at::Tensor& input, at::Tensor& out) {
  auto _input = reinterpret_cast<T_to*>(input.data_ptr<T_from>());
  auto _out = reinterpret_cast<T_to*>(out.data_ptr<T_from>());

  int64_t numel;
  int64_t dim;
  int64_t wg_size;
  int64_t num_group;
  int vec_size;
  get_config<T_to>(input, out, numel, dim, wg_size, num_group, vec_size);

  using accscalar_t = at::opmath_type<T_to>;
  switch (vec_size) {
    VEC_LAUNCH(gelu_erf_mul_dpcpp_functor, 1);
    VEC_LAUNCH(gelu_erf_mul_dpcpp_functor, 2);
    VEC_LAUNCH(gelu_erf_mul_dpcpp_functor, 4);
    VEC_LAUNCH(gelu_erf_mul_dpcpp_functor, 8);
    VEC_LAUNCH(gelu_erf_mul_dpcpp_functor, 16);
    default:
      TORCH_CHECK(false, "Unsupported vector size: ", vec_size);
  }

  return;
}

void gelu_and_mul(at::Tensor& out, at::Tensor& input) {
  input = input.contiguous();
  out = out.contiguous();

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  if (input.scalar_type() == at::ScalarType::Half) {
    gelu_and_mul_sycl<sycl::half, at::Half>(queue, input, out);
  } else {
    gelu_and_mul_sycl<sycl::ext::oneapi::bfloat16, at::BFloat16>(queue, input, out);
  }
  return;
}

