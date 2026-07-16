#include <ATen/ATen.h>
#include <torch/all.h>

#include <limits>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

static constexpr int WG_SIZE = 256;
static constexpr float LOG2E = 1.442695040888963f;
static constexpr int HC = 4;
static constexpr int HC2 = HC * HC;
static constexpr int HC3 = (2 + HC) * HC;
static constexpr int SINKHORN_ITERS = 20;

template <typename scalar_t, int VEC_SIZE, int HC_VAL = 4>
struct HCPostKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int HC2 = HC_VAL * HC_VAL;

  const float* __restrict__ comb_mix;     // [T, HC, HC] FP32 or [T, HC*HC] FP32
  const scalar_t* __restrict__ residual;  // [T, HC, H] BF16
  const float* __restrict__ post_mix;     // [T, HC] FP32
  const scalar_t* __restrict__ x;         // [T, H] BF16
  scalar_t* __restrict__ output;          // [T, HC, H] BF16

  int T_total;
  int hidden_size;
  int HC_val;

  HCPostKernel(
      const float* comb_mix_,
      const scalar_t* residual_,
      const float* post_mix_,
      const scalar_t* x_,
      scalar_t* output_,
      int T_total_,
      int hidden_size_,
      int HC_val_)
      : comb_mix(comb_mix_),
        residual(residual_),
        post_mix(post_mix_),
        x(x_),
        output(output_),
        T_total(T_total_),
        hidden_size(hidden_size_),
        HC_val(HC_val_) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {}

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    const int token_id = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));

    if (token_id >= T_total) return;

    constexpr int kVecSize = VEC_SIZE;
    using vec_t_bf16 = vec_t<scalar_t, kVecSize>;

    const int num_vec_elems = hidden_size / kVecSize;
    const int vec_tail_start = num_vec_elems * kVecSize;
    const int threads_for_h = WG_SIZE;

    float post_local[HC_VAL];
    float comb_local[HC_VAL][HC_VAL];

    if (tid < HC_VAL) {
      post_local[tid] = post_mix[static_cast<int64_t>(token_id) * HC_VAL + tid];
    }

    if (tid < HC_VAL * HC_VAL) {
      const int row = tid / HC_VAL;
      const int col = tid % HC_VAL;
      comb_local[row][col] = comb_mix[static_cast<int64_t>(token_id) * HC_VAL * HC_VAL + tid];
    }

    item.barrier(sycl::access::fence_space::local_space);

    for (int i = tid; i < num_vec_elems; i += threads_for_h) {
      const int h = i * kVecSize;

      vec_t_bf16 x_vec;
      x_vec.load(
          0,
          sycl::multi_ptr<const scalar_t, sycl::access::address_space::global_space>(
              &x[static_cast<int64_t>(token_id) * hidden_size + h]));

      for (int j = 0; j < HC_VAL; j++) {
        float accum[kVecSize] = {};

        float post_val = post_local[j];

#pragma unroll
        for (int v = 0; v < kVecSize; ++v) {
          accum[v] += post_val * static_cast<float>(x_vec[v]);
        }

        for (int k = 0; k < HC_VAL; k++) {
          vec_t_bf16 res_vec;
          res_vec.load(
              0,
              sycl::multi_ptr<const scalar_t, sycl::access::address_space::global_space>(
                  &residual[(static_cast<int64_t>(token_id) * HC_VAL + k) * hidden_size + h]));

          float comb_val = comb_local[k][j];

#pragma unroll
          for (int v = 0; v < kVecSize; ++v) {
            accum[v] += comb_val * static_cast<float>(res_vec[v]);
          }
        }

        vec_t_bf16 out_vec;
#pragma unroll
        for (int v = 0; v < kVecSize; ++v) {
          out_vec[v] = static_cast<scalar_t>(accum[v]);
        }
        out_vec.store(
            0,
            sycl::multi_ptr<scalar_t, sycl::access::address_space::global_space>(
                &output[(static_cast<int64_t>(token_id) * HC_VAL + j) * hidden_size + h]));
      }
    }

    for (int h = vec_tail_start + tid; h < hidden_size; h += threads_for_h) {
      float x_val = static_cast<float>(x[static_cast<int64_t>(token_id) * hidden_size + h]);

      for (int j = 0; j < HC_VAL; j++) {
        float accum = post_local[j] * x_val;

        for (int k = 0; k < HC_VAL; k++) {
          float res_val = static_cast<float>(residual[(static_cast<int64_t>(token_id) * HC_VAL + k) * hidden_size + h]);
          accum += comb_local[k][j] * res_val;
        }

        output[(static_cast<int64_t>(token_id) * HC_VAL + j) * hidden_size + h] = static_cast<scalar_t>(accum);
      }
    }
  }
};

template <int VEC_SIZE>
static void launch_mhc_post_kernel(
    sycl::queue& q,
    const at::Tensor& comb_mix,
    const at::Tensor& residual,
    const at::Tensor& post_mix,
    const at::Tensor& x,
    at::Tensor& output,
    int64_t T,
    int64_t hidden_size,
    int64_t HC_val) {
  using scalar_t = sycl::ext::oneapi::bfloat16;

  auto ker = HCPostKernel<scalar_t, VEC_SIZE>(
      comb_mix.data_ptr<float>(),
      reinterpret_cast<const scalar_t*>(residual.data_ptr<at::BFloat16>()),
      post_mix.data_ptr<float>(),
      reinterpret_cast<const scalar_t*>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<scalar_t*>(output.data_ptr<at::BFloat16>()),
      static_cast<int>(T),
      static_cast<int>(hidden_size),
      static_cast<int>(HC_val));

  sycl_kernel_submit(sycl::range<1>(T * WG_SIZE), sycl::range<1>(WG_SIZE), q, ker);
}

void mhc_post(
    const at::Tensor& comb_mix,
    const at::Tensor& residual,
    const at::Tensor& post_mix,
    const at::Tensor& x,
    at::Tensor& output) {
  CHECK_INPUT(comb_mix);
  CHECK_INPUT(residual);
  CHECK_INPUT(post_mix);
  CHECK_INPUT(x);
  CHECK_INPUT(output);

  c10::DeviceGuard guard(x.device());
  auto q = dpcppGetCurrentQueue();

  int64_t hidden_size = x.size(-1);
  int vec_size = 4;
  if (hidden_size % 16 == 0) {
    vec_size = 16;
  } else if (hidden_size % 8 == 0) {
    vec_size = 8;
  }

  switch (vec_size) {
    case 16:
      launch_mhc_post_kernel<16>(q, comb_mix, residual, post_mix, x, output, x.size(0), hidden_size, residual.size(1));
      break;
    case 8:
      launch_mhc_post_kernel<8>(q, comb_mix, residual, post_mix, x, output, x.size(0), hidden_size, residual.size(1));
      break;
    case 4:
    default:
      launch_mhc_post_kernel<4>(q, comb_mix, residual, post_mix, x, output, x.size(0), hidden_size, residual.size(1));
      break;
  }
}
