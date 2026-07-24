#include <ATen/ATen.h>
#include <torch/all.h>

#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

namespace {

constexpr int SMALL_BATCH_HC = 4;
constexpr int SMALL_BATCH_HC3 = (2 + SMALL_BATCH_HC) * SMALL_BATCH_HC;
constexpr int SMALL_BATCH_WG_SIZE = 128;
constexpr int SMALL_BATCH_SG_SIZE = 16;
constexpr int SMALL_BATCH_NUM_SG = SMALL_BATCH_WG_SIZE / SMALL_BATCH_SG_SIZE;

struct MHCFusedPostPreSmallBatchKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  using bf16_t = sycl::ext::oneapi::bfloat16;

  static constexpr int HC = SMALL_BATCH_HC;
  static constexpr int HC3 = SMALL_BATCH_HC3;
  static constexpr int WG_SIZE = SMALL_BATCH_WG_SIZE;
  static constexpr int SG_SIZE = SMALL_BATCH_SG_SIZE;
  static constexpr int NUM_SG = SMALL_BATCH_NUM_SG;
  static constexpr int SLM_STRIDE = HC3 + 1;

  const bf16_t* __restrict__ x_ptr;
  const bf16_t* __restrict__ residual_ptr;
  const float* __restrict__ post_ptr;
  const float* __restrict__ comb_ptr;
  const float* __restrict__ fn_ptr;
  bf16_t* __restrict__ residual_out_ptr;
  float* __restrict__ mixes_partial_ptr;
  float* __restrict__ sqrsum_partial_ptr;

  int T_total;
  int hidden_size;
  int split_k;
  int hidden_per_split;

  sycl::local_accessor<float, 1> slm_;

  MHCFusedPostPreSmallBatchKernel(
      const bf16_t* x,
      const bf16_t* residual,
      const float* post,
      const float* comb,
      const float* fn,
      bf16_t* residual_out,
      float* mixes_partial,
      float* sqrsum_partial,
      int T_,
      int D_,
      int split_k_)
      : x_ptr(x),
        residual_ptr(residual),
        post_ptr(post),
        comb_ptr(comb),
        fn_ptr(fn),
        residual_out_ptr(residual_out),
        mixes_partial_ptr(mixes_partial),
        sqrsum_partial_ptr(sqrsum_partial),
        T_total(T_),
        hidden_size(D_),
        split_k(split_k_) {
    hidden_per_split = (D_ + split_k_ - 1) / split_k_;
  }

  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm_ = sycl::local_accessor<float, 1>(NUM_SG * SLM_STRIDE, cgh);
  }

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<2> item) const {
    const int token_id = static_cast<int>(item.get_group(0));
    const int split_idx = static_cast<int>(item.get_group(1));
    const int tid = static_cast<int>(item.get_local_id(0));

    if (token_id >= T_total) return;

    sycl::sub_group sg = item.get_sub_group();
    const int sg_id = static_cast<int>(sg.get_group_id()[0]);
    const int lane = static_cast<int>(sg.get_local_id()[0]);

    float post_local[HC];
    float comb_local[HC * HC];
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      post_local[j] = post_ptr[token_id * HC + j];
    }
#pragma unroll
    for (int jk = 0; jk < HC * HC; ++jk) {
      comb_local[jk] = comb_ptr[token_id * HC * HC + jk];
    }

    float mix_acc[HC3];
#pragma unroll
    for (int o = 0; o < HC3; ++o)
      mix_acc[o] = 0.0f;
    float sqrsum_acc = 0.0f;

    const int hidden_start = split_idx * hidden_per_split;
    const int hidden_end = sycl::min(hidden_start + hidden_per_split, hidden_size);

    for (int h = hidden_start + tid; h < hidden_end; h += WG_SIZE) {
      const float x_f = static_cast<float>(x_ptr[static_cast<int64_t>(token_id) * hidden_size + h]);
      float r_f[HC];
#pragma unroll
      for (int k = 0; k < HC; ++k) {
        r_f[k] = static_cast<float>(residual_ptr[(static_cast<int64_t>(token_id) * HC + k) * hidden_size + h]);
      }

      bf16_t cur_res_bf[HC];
      float cur_res_f[HC];
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        float v = post_local[j] * x_f;
#pragma unroll
        for (int k = 0; k < HC; ++k) {
          v += comb_local[k * HC + j] * r_f[k];
        }
        cur_res_bf[j] = static_cast<bf16_t>(v);
        cur_res_f[j] = static_cast<float>(cur_res_bf[j]);
      }

#pragma unroll
      for (int j = 0; j < HC; ++j) {
        residual_out_ptr[(static_cast<int64_t>(token_id) * HC + j) * hidden_size + h] = cur_res_bf[j];
      }

#pragma unroll
      for (int j = 0; j < HC; ++j) {
        sqrsum_acc += cur_res_f[j] * cur_res_f[j];
      }

#pragma unroll
      for (int o = 0; o < HC3; ++o) {
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < HC; ++k) {
          acc += fn_ptr[(static_cast<int64_t>(o) * HC + k) * hidden_size + h] * cur_res_f[k];
        }
        mix_acc[o] += acc;
      }
    }

#pragma unroll
    for (int o = 0; o < HC3; ++o) {
      float v = mix_acc[o];
      v += sycl::permute_group_by_xor(sg, v, 8);
      v += sycl::permute_group_by_xor(sg, v, 4);
      v += sycl::permute_group_by_xor(sg, v, 2);
      v += sycl::permute_group_by_xor(sg, v, 1);
      mix_acc[o] = v;
    }
    {
      float v = sqrsum_acc;
      v += sycl::permute_group_by_xor(sg, v, 8);
      v += sycl::permute_group_by_xor(sg, v, 4);
      v += sycl::permute_group_by_xor(sg, v, 2);
      v += sycl::permute_group_by_xor(sg, v, 1);
      sqrsum_acc = v;
    }

    if (lane == 0) {
#pragma unroll
      for (int o = 0; o < HC3; ++o) {
        slm_[sg_id * SLM_STRIDE + o] = mix_acc[o];
      }
      slm_[sg_id * SLM_STRIDE + HC3] = sqrsum_acc;
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (sg_id == 0) {
      for (int slot = lane; slot < HC3 + 1; slot += SG_SIZE) {
        float sum = 0.0f;
#pragma unroll
        for (int s = 0; s < NUM_SG; ++s) {
          sum += slm_[s * SLM_STRIDE + slot];
        }
        if (slot < HC3) {
          const int64_t idx = (static_cast<int64_t>(split_idx) * T_total + token_id) * HC3 + slot;
          mixes_partial_ptr[idx] = sum;
        } else {
          const int64_t idx = static_cast<int64_t>(split_idx) * T_total + token_id;
          sqrsum_partial_ptr[idx] = sum;
        }
      }
    }
  }
};

}  // namespace

void mhc_fused_post_pre_small_batch(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_layer_mix_2d,
    const at::Tensor& comb_res_mix_3d,
    const at::Tensor& fn,
    at::Tensor& residual_out,
    at::Tensor& mixes_partial_out,
    at::Tensor& sqrsum_partial_out,
    int64_t split_k) {
  using bf16_t = sycl::ext::oneapi::bfloat16;

  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");
  TORCH_CHECK(post_layer_mix_2d.scalar_type() == at::kFloat, "post_layer_mix must be float32");
  TORCH_CHECK(comb_res_mix_3d.scalar_type() == at::kFloat, "comb_res_mix must be float32");
  TORCH_CHECK(fn.scalar_type() == at::kFloat, "fn must be float32");
  TORCH_CHECK(residual_out.scalar_type() == at::kBFloat16, "residual_out must be bfloat16");
  TORCH_CHECK(mixes_partial_out.scalar_type() == at::kFloat, "mixes_partial_out must be float32");
  TORCH_CHECK(sqrsum_partial_out.scalar_type() == at::kFloat, "sqrsum_partial_out must be float32");

  const int64_t T = x.size(0);
  const int64_t D = x.size(1);

  TORCH_CHECK(
      residual.size(0) == T && residual.size(1) == SMALL_BATCH_HC && residual.size(2) == D,
      "residual shape mismatch in small-batch fused kernel");
  TORCH_CHECK(
      post_layer_mix_2d.size(0) == T && post_layer_mix_2d.size(1) == SMALL_BATCH_HC, "post_layer_mix shape mismatch");
  TORCH_CHECK(
      comb_res_mix_3d.size(0) == T && comb_res_mix_3d.size(1) == SMALL_BATCH_HC &&
          comb_res_mix_3d.size(2) == SMALL_BATCH_HC,
      "comb_res_mix shape mismatch");
  TORCH_CHECK(fn.dim() == 2 && fn.size(0) == SMALL_BATCH_HC3 && fn.size(1) == SMALL_BATCH_HC * D, "fn shape mismatch");
  TORCH_CHECK(residual_out.sizes() == residual.sizes(), "residual_out shape mismatch");
  TORCH_CHECK(
      mixes_partial_out.dim() == 3 && mixes_partial_out.size(0) == split_k && mixes_partial_out.size(1) == T &&
          mixes_partial_out.size(2) == SMALL_BATCH_HC3,
      "mixes_partial_out shape mismatch");
  TORCH_CHECK(
      sqrsum_partial_out.dim() == 2 && sqrsum_partial_out.size(0) == split_k && sqrsum_partial_out.size(1) == T,
      "sqrsum_partial_out shape mismatch");

  if (T == 0 || split_k == 0) return;

  auto q = dpcppGetCurrentQueue();

  MHCFusedPostPreSmallBatchKernel ker(
      reinterpret_cast<const bf16_t*>(x.data_ptr<at::BFloat16>()),
      reinterpret_cast<const bf16_t*>(residual.data_ptr<at::BFloat16>()),
      post_layer_mix_2d.data_ptr<float>(),
      comb_res_mix_3d.data_ptr<float>(),
      fn.data_ptr<float>(),
      reinterpret_cast<bf16_t*>(residual_out.data_ptr<at::BFloat16>()),
      mixes_partial_out.data_ptr<float>(),
      sqrsum_partial_out.data_ptr<float>(),
      static_cast<int>(T),
      static_cast<int>(D),
      static_cast<int>(split_k));

  sycl::range<2> global(static_cast<size_t>(T) * SMALL_BATCH_WG_SIZE, static_cast<size_t>(split_k));
  sycl::range<2> local(SMALL_BATCH_WG_SIZE, 1);
  sycl_kernel_submit(global, local, q, ker);
}
