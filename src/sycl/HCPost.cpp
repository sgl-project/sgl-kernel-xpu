#include <ATen/ATen.h>
#include <torch/all.h>

#include <limits>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

static constexpr int WG_SIZE = 256;
static constexpr int D_BLOCK = 1024;  // Process D in 1024-element blocks
static constexpr int HC = 4;

template <typename scalar_t, int VEC_SIZE, int HC_MULT = 4>
struct HCPostKernel {
  const scalar_t* __restrict__ x;         // [T, D]     bf16
  const scalar_t* __restrict__ residual;  // [T, HC, D] bf16
  const float* __restrict__ post;         // [T, HC]    fp32
  const float* __restrict__ comb;         // [T, HC, HC] fp32

  scalar_t* __restrict__ out;  // [T, HC, D] bf16

  int T;
  int D;

  HCPostKernel(
      const scalar_t* x_,
      const scalar_t* residual_,
      const float* post_,
      const float* comb_,
      scalar_t* out_,
      int T_,
      int D_)
      : x(x_), residual(residual_), post(post_), comb(comb_), out(out_), T(T_), D(D_) {}

  void operator()(sycl::nd_item<2> item) const {
    uint32_t token_id = item.get_group(0);
    uint32_t d_block_id = item.get_group(1);
    uint32_t local_id = item.get_local_id(0);

    if (token_id >= T) return;

    int d_start = d_block_id * D_BLOCK;
    int d_end = sycl::min(d_start + D_BLOCK, D);

    float post_local[HC_MULT];
#pragma unroll
    for (int i = 0; i < HC_MULT; i++) {
      post_local[i] = post[token_id * HC_MULT + i];
    }

    float comb_local[HC_MULT * HC_MULT];
#pragma unroll
    for (int i = 0; i < HC_MULT * HC_MULT; i++) {
      comb_local[i] = comb[token_id * HC_MULT * HC_MULT + i];
    }

    constexpr int kVecSize = VEC_SIZE;
    using vec_in = vec_t<scalar_t, kVecSize>;
    using vec_out = vec_t<scalar_t, kVecSize>;

    for (int d = d_start + local_id * kVecSize; d < d_end; d += WG_SIZE * kVecSize) {
      vec_in x_vec;
      x_vec.load(0, sycl::multi_ptr<const scalar_t, sycl::access::address_space::global_space>(&x[token_id * D + d]));

      float x_fp32[kVecSize];
#pragma unroll
      for (int i = 0; i < kVecSize; i++) {
        x_fp32[i] = x_vec[i];
      }

      float residual_fp32[HC_MULT][kVecSize];
#pragma unroll
      for (int k = 0; k < HC_MULT; k++) {
        vec_in residual_vec;
        residual_vec.load(
            0,
            sycl::multi_ptr<const scalar_t, sycl::access::address_space::global_space>(
                &residual[(token_id * HC_MULT + k) * D + d]));
#pragma unroll
        for (int i = 0; i < kVecSize; i++) {
          residual_fp32[k][i] = residual_vec[i];
        }
      }

      // out[t, j, d] = post[t,j] * x[t,d] + Σ_i comb[t,i,j] * residual[t,i,d]
#pragma unroll
      for (int j = 0; j < HC_MULT; j++) {
        vec_out out_vec;

#pragma unroll
        for (int i = 0; i < kVecSize; i++) {
          float accum = post_local[j] * x_fp32[i];

#pragma unroll
          for (int k = 0; k < HC_MULT; k++) {
            accum += comb_local[k * HC_MULT + j] * residual_fp32[k][i];
          }

          out_vec[i] = accum;
        }

        out_vec.store(
            0,
            sycl::multi_ptr<scalar_t, sycl::access::address_space::global_space>(
                &out[(token_id * HC_MULT + j) * D + d]));
      }
    }
  }
};

template <int VEC_SIZE>
static void launch_hc_post_kernel(
    sycl::queue& q,
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post,
    const at::Tensor& comb,
    at::Tensor& out,
    int64_t T,
    int64_t D) {
  using scalar_t = sycl::ext::oneapi::bfloat16;

  auto ker = HCPostKernel<scalar_t, VEC_SIZE>(
      reinterpret_cast<const scalar_t*>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<const scalar_t*>(residual.data_ptr<at::BFloat16>()),
      post.data_ptr<float>(),
      comb.data_ptr<float>(),
      reinterpret_cast<scalar_t*>(out.data_ptr<at::BFloat16>()),
      static_cast<int>(T),
      static_cast<int>(D));

  // Grid = (T, ceil_div(D, D_BLOCK))
  int64_t grid_x = T;
  int64_t grid_y = (D + D_BLOCK - 1) / D_BLOCK;

  sycl_kernel_submit(sycl::range<2>(grid_x * WG_SIZE, grid_y), sycl::range<2>(WG_SIZE, 1), q, ker);
}

void hc_post(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_layer_mix,
    const at::Tensor& comb_res_mix,
    at::Tensor& out) {
  CHECK_INPUT(x);
  CHECK_INPUT(residual);
  CHECK_INPUT(post_layer_mix);
  CHECK_INPUT(comb_res_mix);
  CHECK_INPUT(out);

  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");
  TORCH_CHECK(post_layer_mix.scalar_type() == at::kFloat, "post_layer_mix must be float32");
  TORCH_CHECK(comb_res_mix.scalar_type() == at::kFloat, "comb_res_mix must be float32");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bfloat16");

  int64_t T = x.size(0);
  int64_t D = x.size(1);

  TORCH_CHECK(residual.dim() == 3, "residual must be 3D [T, HC, D]");
  TORCH_CHECK(residual.size(0) == T, "residual T mismatch");
  TORCH_CHECK(residual.size(1) == 4, "residual must have 4 channels (HC=4)");
  TORCH_CHECK(residual.size(2) == D, "residual D mismatch");

  TORCH_CHECK(post_layer_mix.dim() == 2, "post_layer_mix must be 2D [T, HC]");
  TORCH_CHECK(post_layer_mix.size(0) == T, "post_layer_mix T mismatch");
  TORCH_CHECK(post_layer_mix.size(1) == 4, "post_layer_mix must have 4 elements (HC=4)");

  TORCH_CHECK(comb_res_mix.dim() == 3, "comb_res_mix must be 3D [T, HC, HC]");
  TORCH_CHECK(comb_res_mix.size(0) == T, "comb_res_mix T mismatch");
  TORCH_CHECK(comb_res_mix.size(1) == 4, "comb_res_mix must have 4 rows (HC=4)");
  TORCH_CHECK(comb_res_mix.size(2) == 4, "comb_res_mix must have 4 cols (HC=4)");

  TORCH_CHECK(out.dim() == 3, "out must be 3D [T, HC, D]");
  TORCH_CHECK(out.size(0) == T, "out T mismatch");
  TORCH_CHECK(out.size(1) == 4, "out must have 4 channels (HC=4)");
  TORCH_CHECK(out.size(2) == D, "out D mismatch");

  auto q = dpcppGetCurrentQueue();

  constexpr int VEC_SIZE = 4;
  TORCH_CHECK(D % VEC_SIZE == 0, "D must be a multiple of VEC_SIZE (", VEC_SIZE, "), got D=", D);
  launch_hc_post_kernel<VEC_SIZE>(q, x, residual, post_layer_mix, comb_res_mix, out, T, D);
}
