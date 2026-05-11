#include <ATen/ATen.h>
#include <torch/all.h>

#include <limits>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

static constexpr int WG_SIZE = 256;  // threads per WG; tokens_per_wg = WG_SIZE / (HC*HC)
static constexpr int HC = 4;
static constexpr int SINKHORN_ITERS = 20;

struct HCSplitSinkhornKernel {
  static constexpr int HC2 = HC * HC;
  static constexpr int COL_SIZE = (2 + HC) * HC;

  const float* __restrict__ mixes;     // [T, COL_SIZE]
  const float* __restrict__ hc_scale;  // [3]
  const float* __restrict__ hc_base;   // [COL_SIZE]
  float* __restrict__ pre;             // [T, HC]
  float* __restrict__ post;            // [T, HC]
  float* __restrict__ comb;            // [T, HC, HC]
  int T_total;
  float eps;

  [[sycl::reqd_sub_group_size(HC2)]] void operator()(sycl::nd_item<1> item) const {
    sycl::sub_group sg = item.get_sub_group();

    constexpr int tokens_per_wg = WG_SIZE / HC2;
    const int token_id = static_cast<int>(item.get_group(0)) * tokens_per_wg + static_cast<int>(sg.get_group_id()[0]);
    const int tid = static_cast<int>(sg.get_local_id()[0]);  // 0..HC*HC-1

    if (token_id >= T_total) return;

    const int row_i = tid / HC;
    const int col_j = tid % HC;

    const float scale0 = hc_scale[0];
    const float scale1 = hc_scale[1];
    const float scale2 = hc_scale[2];

    const float* row = mixes + static_cast<int64_t>(token_id) * COL_SIZE;

    if (tid < 2 * HC) {
      if (row_i == 0) {
        const float pre_logit = row[col_j] * scale0 + hc_base[col_j];
        pre[static_cast<int64_t>(token_id) * HC + col_j] = 1.0f / (1.0f + sycl::exp(-pre_logit)) + eps;
      } else {
        const float post_logit = row[HC + col_j] * scale1 + hc_base[HC + col_j];
        post[static_cast<int64_t>(token_id) * HC + col_j] = 2.0f / (1.0f + sycl::exp(-post_logit));
      }
    }

    float comb_logit = row[2 * HC + tid] * scale2 + hc_base[2 * HC + tid];

    float row_max = comb_logit;
#pragma unroll
    for (int mask = 1; mask < HC; mask <<= 1)
      row_max = sycl::fmax(row_max, sycl::permute_group_by_xor(sg, row_max, mask));

    float comb_val = sycl::exp(comb_logit - row_max);
    float row_sum = comb_val;
#pragma unroll
    for (int mask = 1; mask < HC; mask <<= 1)
      row_sum += sycl::permute_group_by_xor(sg, row_sum, mask);

    comb_val = comb_val / row_sum + eps;

    float col_sum = comb_val;
#pragma unroll
    for (int mask = HC; mask < HC2; mask <<= 1)
      col_sum += sycl::permute_group_by_xor(sg, col_sum, mask);
    comb_val /= col_sum + eps;

#pragma unroll
    for (int iter = 1; iter < SINKHORN_ITERS; ++iter) {
      row_sum = comb_val;
#pragma unroll
      for (int mask = 1; mask < HC; mask <<= 1)
        row_sum += sycl::permute_group_by_xor(sg, row_sum, mask);
      comb_val /= row_sum + eps;

      col_sum = comb_val;
#pragma unroll
      for (int mask = HC; mask < HC2; mask <<= 1)
        col_sum += sycl::permute_group_by_xor(sg, col_sum, mask);
      comb_val /= col_sum + eps;
    }

    comb[static_cast<int64_t>(token_id) * HC2 + tid] = comb_val;
  }
};

void hc_split_sinkhorn(
    const at::Tensor& mixes,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    at::Tensor& pre,
    at::Tensor& post,
    at::Tensor& comb,
    int64_t hc_mult,
    int64_t sinkhorn_iters,
    double eps) {
  CHECK_INPUT(mixes);
  CHECK_INPUT(hc_scale);
  CHECK_INPUT(hc_base);
  CHECK_INPUT(pre);
  CHECK_INPUT(post);
  CHECK_INPUT(comb);

  TORCH_CHECK(mixes.scalar_type() == at::kFloat, "mixes must be float32");
  TORCH_CHECK(hc_scale.scalar_type() == at::kFloat, "hc_scale must be float32");
  TORCH_CHECK(hc_base.scalar_type() == at::kFloat, "hc_base must be float32");
  TORCH_CHECK(pre.scalar_type() == at::kFloat, "pre must be float32");
  TORCH_CHECK(post.scalar_type() == at::kFloat, "post must be float32");
  TORCH_CHECK(comb.scalar_type() == at::kFloat, "comb must be float32");

  constexpr int col_size = HCSplitSinkhornKernel::COL_SIZE;

  TORCH_CHECK(static_cast<int>(hc_mult) == HC, "hc_mult must be ", HC, ", got ", hc_mult);
  TORCH_CHECK(
      static_cast<int>(sinkhorn_iters) == SINKHORN_ITERS,
      "sinkhorn_iters must be ",
      SINKHORN_ITERS,
      ", got ",
      sinkhorn_iters);
  TORCH_CHECK(mixes.size(-1) == col_size, "mixes last dim must be (2+HC)*HC=", col_size, ", got ", mixes.size(-1));
  TORCH_CHECK(hc_scale.numel() == 3, "hc_scale must have 3 elements");
  TORCH_CHECK(hc_base.numel() == col_size, "hc_base must have (2+HC)*HC=", col_size, " elements");

  const int64_t T = mixes.numel() / col_size;
  TORCH_CHECK(T < std::numeric_limits<int>::max(), "T (", T, ") must fit in int32");

  TORCH_CHECK(pre.numel() == T * HC, "pre must have T*HC=", T * HC, " elements, got ", pre.numel());
  TORCH_CHECK(post.numel() == T * HC, "post must have T*HC=", T * HC, " elements, got ", post.numel());
  TORCH_CHECK(comb.numel() == T * HC * HC, "comb must have T*HC*HC=", T * HC * HC, " elements, got ", comb.numel());

  auto q = dpcppGetCurrentQueue();

  constexpr int tokens_per_wg = WG_SIZE / (HC * HC);
  const int64_t num_wg = (T + tokens_per_wg - 1) / tokens_per_wg;
  auto ker = HCSplitSinkhornKernel{
      mixes.data_ptr<float>(),
      hc_scale.data_ptr<float>(),
      hc_base.data_ptr<float>(),
      pre.data_ptr<float>(),
      post.data_ptr<float>(),
      comb.data_ptr<float>(),
      static_cast<int>(T),
      static_cast<float>(eps),
  };
  sycl_kernel_submit(num_wg * WG_SIZE, static_cast<int64_t>(WG_SIZE), q, ker);
}
