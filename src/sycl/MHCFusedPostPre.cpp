#include <ATen/ATen.h>

#include <algorithm>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "sgl_kernel_ops.h"

void mhc_fused_post_pre_small_batch(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_layer_mix_2d,
    const at::Tensor& comb_res_mix_3d,
    const at::Tensor& fn,
    at::Tensor& residual_out,
    at::Tensor& mixes_partial_out,
    at::Tensor& sqrsum_partial_out,
    int64_t split_k);

namespace {

constexpr int64_t kSmallBatchThreshold = 32;

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

}  // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> mhc_fused_post_pre(
    const at::Tensor& x,
    const at::Tensor& residual,
    const at::Tensor& post_layer_mix,
    const at::Tensor& comb_res_mix,
    const at::Tensor& fn,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    double rms_eps,
    double hc_pre_eps,
    double hc_sinkhorn_eps,
    double hc_post_mult_value,
    int64_t sinkhorn_repeat,
    int64_t n_splits,
    std::optional<at::Tensor> norm_weight,
    std::optional<double> norm_eps) {
  CHECK_INPUT(x);
  CHECK_INPUT(residual);
  CHECK_INPUT(post_layer_mix);
  CHECK_INPUT(comb_res_mix);
  CHECK_INPUT(fn);
  CHECK_INPUT(hc_scale);
  CHECK_INPUT(hc_base);

  TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
  TORCH_CHECK(residual.scalar_type() == at::kBFloat16, "residual must be bfloat16");
  TORCH_CHECK(post_layer_mix.scalar_type() == at::kFloat, "post_layer_mix must be float32");
  TORCH_CHECK(comb_res_mix.scalar_type() == at::kFloat, "comb_res_mix must be float32");
  TORCH_CHECK(fn.scalar_type() == at::kFloat, "fn must be float32");
  TORCH_CHECK(hc_scale.scalar_type() == at::kFloat, "hc_scale must be float32");
  TORCH_CHECK(hc_base.scalar_type() == at::kFloat, "hc_base must be float32");

  TORCH_CHECK(x.dim() == 2, "x must be 2D [T, D]");
  TORCH_CHECK(residual.dim() == 3, "residual must be 3D [T, HC, D]");

  const int64_t t = x.size(0);
  const int64_t hidden_size = x.size(1);
  const int64_t hc_mult = residual.size(1);
  const int64_t hc_mult3 = (2 + hc_mult) * hc_mult;
  const int64_t hc_hidden = hc_mult * hidden_size;

  TORCH_CHECK(residual.size(0) == t, "residual T mismatch");
  TORCH_CHECK(residual.size(2) == hidden_size, "residual D mismatch");
  TORCH_CHECK(hc_mult == 4, "mhc_fused_post_pre currently supports only HC=4");
  TORCH_CHECK(sinkhorn_repeat == 20, "mhc_fused_post_pre currently supports only sinkhorn_repeat=20");
  TORCH_CHECK(fn.dim() == 2, "fn must be 2D [HC3, HC*D]");
  TORCH_CHECK(fn.size(0) == hc_mult3, "fn row mismatch");
  TORCH_CHECK(fn.size(1) == hc_hidden, "fn column mismatch");
  TORCH_CHECK(hc_scale.numel() == 3, "hc_scale must have 3 elements");
  TORCH_CHECK(hc_base.numel() == hc_mult3, "hc_base size mismatch");

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

  int64_t n_splits_pre = choose_n_splits(t, hc_hidden, n_splits);

  at::Tensor residual_cur = at::empty_like(residual);

  at::Tensor gemm_out_mul;
  at::Tensor gemm_out_sqrsum;

  if (t <= kSmallBatchThreshold && t > 0) {
    gemm_out_mul = at::empty({n_splits_pre, t, hc_mult3}, residual.options().dtype(at::kFloat));
    gemm_out_sqrsum = at::empty({n_splits_pre, t}, residual.options().dtype(at::kFloat));
    mhc_fused_post_pre_small_batch(
        x, residual, post_2d, comb_3d, fn, residual_cur, gemm_out_mul, gemm_out_sqrsum, n_splits_pre);
  } else {
    hc_post(x, residual, post_2d, comb_3d, residual_cur);

    gemm_out_mul = at::empty({n_splits_pre, t, hc_mult3}, residual.options().dtype(at::kFloat));
    gemm_out_sqrsum = at::empty({n_splits_pre, t}, residual.options().dtype(at::kFloat));
    at::Tensor a = residual_cur.reshape({t, hc_hidden});
    hc_pre_gemm_sqr_sum(gemm_out_mul, gemm_out_sqrsum, a, fn);
  }

  at::Tensor post_mix_cur = at::empty({t, hc_mult}, residual.options().dtype(at::kFloat));
  at::Tensor comb_mix_cur = at::empty({t, hc_mult, hc_mult}, residual.options().dtype(at::kFloat));
  at::Tensor layer_input_cur = at::empty({t, hidden_size}, residual.options());

  hc_pre_big_fuse(
      gemm_out_mul,
      gemm_out_sqrsum,
      hc_scale,
      hc_base,
      residual_cur,
      post_mix_cur,
      comb_mix_cur,
      layer_input_cur,
      hc_mult,
      sinkhorn_repeat,
      n_splits_pre,
      rms_eps,
      hc_pre_eps,
      hc_sinkhorn_eps,
      hc_post_mult_value,
      norm_weight,
      norm_eps);

  return {
      residual_cur,
      post_mix_cur.unsqueeze(-1),
      comb_mix_cur,
      layer_input_cur,
  };
}
