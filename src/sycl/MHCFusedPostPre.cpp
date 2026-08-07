#include <ATen/ATen.h>
#include <torch/all.h>

#include <algorithm>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_ops.h"

namespace {

constexpr int SMALL_BATCH_HC = 4;
constexpr int SMALL_BATCH_HC3 = (2 + SMALL_BATCH_HC) * SMALL_BATCH_HC;
constexpr int SMALL_BATCH_WG_SIZE = 128;
constexpr int SMALL_BATCH_SG_SIZE = 16;
constexpr int SMALL_BATCH_NUM_SG = SMALL_BATCH_WG_SIZE / SMALL_BATCH_SG_SIZE;

constexpr int64_t kSmallBatchThreshold = 32;

struct MHCFusedPostPreFmaKernel : public __SYCL_KER_CONFIG_CONVENTION__ {
  using bf16_t = sycl::ext::oneapi::bfloat16;

  static constexpr int HC = SMALL_BATCH_HC;
  static constexpr int HC3 = SMALL_BATCH_HC3;
  static constexpr int WG_SIZE = SMALL_BATCH_WG_SIZE;
  static constexpr int SG_SIZE = SMALL_BATCH_SG_SIZE;
  static constexpr int NUM_SG = SMALL_BATCH_NUM_SG;
  static constexpr int VEC_SIZE = 4;
  static constexpr int SLM_STRIDE = HC3 + 1;
  static constexpr int SLM_PARTIAL_SIZE = NUM_SG * SLM_STRIDE;
  static constexpr int SLM_POST_OFFSET = SLM_PARTIAL_SIZE;
  static constexpr int SLM_COMB_OFFSET = SLM_POST_OFFSET + HC;

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

  MHCFusedPostPreFmaKernel(
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
    slm_ = sycl::local_accessor<float, 1>(SLM_COMB_OFFSET + HC * HC, cgh);
  }

  [[sycl::reqd_sub_group_size(SG_SIZE)]] void operator()(sycl::nd_item<2> item) const {
    const int token_id = static_cast<int>(item.get_group(0));
    const int split_idx = static_cast<int>(item.get_group(1));
    const int tid = static_cast<int>(item.get_local_id(0));

    if (token_id >= T_total) return;

    sycl::sub_group sg = item.get_sub_group();
    const int sg_id = static_cast<int>(sg.get_group_id()[0]);
    const int lane = static_cast<int>(sg.get_local_id()[0]);

    if (tid < HC) {
      slm_[SLM_POST_OFFSET + tid] = post_ptr[token_id * HC + tid];
    }
    if (tid < HC * HC) {
      slm_[SLM_COMB_OFFSET + tid] = comb_ptr[token_id * HC * HC + tid];
    }
    item.barrier(sycl::access::fence_space::local_space);

    float post_local[HC];
    float comb_local[HC * HC];
#pragma unroll
    for (int j = 0; j < HC; ++j) {
      post_local[j] = slm_[SLM_POST_OFFSET + j];
    }
#pragma unroll
    for (int jk = 0; jk < HC * HC; ++jk) {
      comb_local[jk] = slm_[SLM_COMB_OFFSET + jk];
    }

    float mix_acc[HC3];
#pragma unroll
    for (int o = 0; o < HC3; ++o)
      mix_acc[o] = 0.0f;
    float sqrsum_acc = 0.0f;

    const int hidden_start = split_idx * hidden_per_split;
    const int hidden_end = sycl::min(hidden_start + hidden_per_split, hidden_size);

    using vec_bf16 = vec_t<bf16_t, VEC_SIZE>;
    const int hidden_count = hidden_end - hidden_start;
    const int num_vec = hidden_count / VEC_SIZE;
    const int vec_tail_start = hidden_start + num_vec * VEC_SIZE;

    for (int vec_idx = tid; vec_idx < num_vec; vec_idx += WG_SIZE) {
      const int h = hidden_start + vec_idx * VEC_SIZE;
      vec_bf16 x_vec;
      x_vec.load(
          0,
          sycl::multi_ptr<const bf16_t, sycl::access::address_space::global_space>(
              &x_ptr[static_cast<int64_t>(token_id) * hidden_size + h]));

      float x_f[VEC_SIZE];
#pragma unroll
      for (int i = 0; i < VEC_SIZE; ++i) {
        x_f[i] = static_cast<float>(x_vec[i]);
      }

      float r_f[HC][VEC_SIZE];
#pragma unroll
      for (int k = 0; k < HC; ++k) {
        vec_bf16 r_vec;
        r_vec.load(
            0,
            sycl::multi_ptr<const bf16_t, sycl::access::address_space::global_space>(
                &residual_ptr[(static_cast<int64_t>(token_id) * HC + k) * hidden_size + h]));
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          r_f[k][i] = static_cast<float>(r_vec[i]);
        }
      }

      bf16_t cur_res_bf[HC][VEC_SIZE];
      float cur_res_f[HC][VEC_SIZE];
#pragma unroll
      for (int j = 0; j < HC; ++j) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          float v = post_local[j] * x_f[i];
#pragma unroll
          for (int k = 0; k < HC; ++k) {
            v += comb_local[k * HC + j] * r_f[k][i];
          }
          cur_res_bf[j][i] = static_cast<bf16_t>(v);
          cur_res_f[j][i] = static_cast<float>(cur_res_bf[j][i]);
        }
      }

#pragma unroll
      for (int j = 0; j < HC; ++j) {
        vec_bf16 out_vec;
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          out_vec[i] = cur_res_bf[j][i];
        }
        out_vec.store(
            0,
            sycl::multi_ptr<bf16_t, sycl::access::address_space::global_space>(
                &residual_out_ptr[(static_cast<int64_t>(token_id) * HC + j) * hidden_size + h]));
      }

#pragma unroll
      for (int j = 0; j < HC; ++j) {
#pragma unroll
        for (int i = 0; i < VEC_SIZE; ++i) {
          sqrsum_acc += cur_res_f[j][i] * cur_res_f[j][i];
        }
      }

#pragma unroll
      for (int o = 0; o < HC3; ++o) {
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < HC; ++k) {
#pragma unroll
          for (int i = 0; i < VEC_SIZE; ++i) {
            acc += fn_ptr[(static_cast<int64_t>(o) * HC + k) * hidden_size + (h + i)] * cur_res_f[k][i];
          }
        }
        mix_acc[o] += acc;
      }
    }

    for (int h = vec_tail_start + tid; h < hidden_end; h += WG_SIZE) {
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

inline int64_t choose_small_batch_split_k(int64_t t) {
  if (t <= 4) return 32;
  if (t <= 16) return 8;
  return 4;
}

inline int64_t choose_large_batch_n_splits(int64_t t, int64_t) {
  return t <= 2048 ? 32 : 1;
}

inline int64_t choose_n_splits(int64_t t, int64_t hc_hidden, int64_t n_splits_hint) {
  if (n_splits_hint > 0) {
    return n_splits_hint;
  }
  if (t <= kSmallBatchThreshold) {
    return choose_small_batch_split_k(t);
  }
  return choose_large_batch_n_splits(t, hc_hidden);
}

void mhc_fused_post_pre_fma_kernel(
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

  MHCFusedPostPreFmaKernel ker(
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

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> mhc_fused_post_pre_fma(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_layer_mix,
    const at::Tensor& comb_res_mix,
    const at::Tensor& fn,
    int64_t n_splits) {
  CHECK_INPUT(x);
  CHECK_INPUT(residual);
  CHECK_INPUT(post_layer_mix);
  CHECK_INPUT(comb_res_mix);
  CHECK_INPUT(fn);

  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");
  TORCH_CHECK(post_layer_mix.scalar_type() == at::kFloat, "post_layer_mix must be float32");
  TORCH_CHECK(comb_res_mix.scalar_type() == at::kFloat, "comb_res_mix must be float32");
  TORCH_CHECK(fn.scalar_type() == at::kFloat, "fn must be float32");

  TORCH_CHECK(x.dim() == 2, "x must be 2D [T, D]");
  TORCH_CHECK(residual.dim() == 3, "residual must be 3D [T, HC, D]");

  const int64_t t = x.size(0);
  const int64_t hidden_size = x.size(1);
  const int64_t hc_mult = residual.size(1);
  const int64_t hc_mult3 = (2 + hc_mult) * hc_mult;
  const int64_t hc_hidden = hc_mult * hidden_size;

  TORCH_CHECK(residual.size(0) == t, "residual T mismatch");
  TORCH_CHECK(residual.size(2) == hidden_size, "residual D mismatch");
  TORCH_CHECK(hc_mult == 4, "mhc_fused_post_pre_fma currently supports only HC=4");
  TORCH_CHECK(fn.dim() == 2, "fn must be 2D [HC3, HC*D]");
  TORCH_CHECK(fn.size(0) == hc_mult3, "fn row mismatch");
  TORCH_CHECK(fn.size(1) == hc_hidden, "fn column mismatch");

  at::Tensor post_2d = post_layer_mix;
  if (post_2d.dim() == 3) {
    TORCH_CHECK(post_2d.size(2) == 1, "post_layer_mix last dim must be 1 when rank=3");
    post_2d = post_2d.squeeze(-1);
  }
  TORCH_CHECK(post_2d.dim() == 2, "post_layer_mix must be [T, HC] or [T, HC, 1]");
  TORCH_CHECK(post_2d.size(0) == t && post_2d.size(1) == hc_mult, "post_layer_mix shape mismatch");

  at::Tensor comb_3d = comb_res_mix;
  if (comb_3d.dim() == 2) {
    TORCH_CHECK(comb_3d.size(1) == hc_mult * hc_mult, "comb_res_mix rank-2 shape mismatch");
    comb_3d = comb_3d.view({t, hc_mult, hc_mult});
  }
  TORCH_CHECK(comb_3d.dim() == 3, "comb_res_mix must be [T, HC, HC] or [T, HC*HC]");
  TORCH_CHECK(
      comb_3d.size(0) == t && comb_3d.size(1) == hc_mult && comb_3d.size(2) == hc_mult, "comb_res_mix shape mismatch");

  const int64_t n_splits_pre = choose_n_splits(t, hc_hidden, n_splits);
  at::Tensor residual_cur = at::empty_like(residual);
  at::Tensor gemm_out_mul = at::empty({n_splits_pre, t, hc_mult3}, residual.options().dtype(at::kFloat));
  at::Tensor gemm_out_sqrsum = at::empty({n_splits_pre, t}, residual.options().dtype(at::kFloat));

  if (t > 0) {
    mhc_fused_post_pre_fma_kernel(
        x, residual, post_2d, comb_3d, fn, residual_cur, gemm_out_mul, gemm_out_sqrsum, n_splits_pre);
  }

  return {residual_cur, gemm_out_mul, gemm_out_sqrsum};
}
