#include <ATen/ATen.h>
#include <torch/all.h>

#include <limits>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

#define DPCPP_KER_PRINTF sycl::ext::oneapi::experimental::printf

// Kernel configuration
static constexpr int WG_SIZE = 128;        // 8 subgroups of 16 threads each
static constexpr int HC = 4;               // hc_mult value
static constexpr int HC2 = HC * HC;        // 16
static constexpr int HC3 = (2 + HC) * HC;  // 24

static constexpr float LOG2E = 1.442695040888963f;  // log2(e) for exp2 conversion

template <typename scalar_t>
struct HCPreBigFuseKernel {
  const float* __restrict__ gemm_out_mul;     // [n_splits, T, 24] FP32
  const float* __restrict__ gemm_out_sqrsum;  // [n_splits, T] FP32
  const float* __restrict__ hc_scale;         // [3] FP32
  const float* __restrict__ hc_base;          // [24] FP32
  const scalar_t* __restrict__ residual;      // [T, 4, D] BF16
  float* __restrict__ post_mix;               // [T, 4] FP32
  float* __restrict__ comb_mix;               // [T, 16] FP32
  scalar_t* __restrict__ layer_input;         // [T, D] BF16

  int T_total;
  int hidden_size;
  int n_splits;
  float rms_eps;
  float hc_pre_eps;
  float hc_sinkhorn_eps;
  float hc_post_mult_value;
  int sinkhorn_iters;

  sycl::local_accessor<float, 1> slm_;

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    sycl::sub_group sg = item.get_sub_group();

    const int token_id = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));      // 0..127
    const int sg_id = static_cast<int>(sg.get_group_id()[0]);    // 0..7
    const int lane_id = static_cast<int>(sg.get_local_id()[0]);  // 0..15

    if (token_id >= T_total) return;

    // Access shared memory
    float* mixes_shared = slm_.get_multi_ptr<sycl::access::decorated::no>().get();  // 24 floats
    float* pre_mix_shared = mixes_shared + HC3;                                     // 4 floats

    // RMS normalization fused with mix accumulation
    if (tid < HC3) {
      float sqrsum = 0.0f;
      float mix_val = 0.0f;
      for (int split = 0; split < n_splits; split++) {
        const int row = split * T_total + token_id;
        sqrsum += gemm_out_sqrsum[row];
        mix_val += gemm_out_mul[row * HC3 + tid];
      }
      const float rms = sycl::native::rsqrt(sqrsum * sycl::native::recip(HC * hidden_size) + rms_eps);
      mixes_shared[tid] = mix_val * rms;
    }

    item.barrier(sycl::access::fence_space::local_space);

    if (sg_id < 2) {
      // Threads 0-31 (Subgroups 0-1)
      if (sg_id == 0) {
        // post_mix from mixes[4:8]
        if (lane_id < HC) {
          const float post_logit = mixes_shared[HC + lane_id] * hc_scale[1] + hc_base[HC + lane_id];
          post_mix[static_cast<int64_t>(token_id) * HC + lane_id] =
              hc_post_mult_value * sycl::native::recip(1.0f + sycl::native::exp2(-post_logit * LOG2E));
        }

        // sinkhorn from mixes[8:24]
        float comb_logit = mixes_shared[2 * HC + lane_id] * hc_scale[2] + hc_base[2 * HC + lane_id];

        float row_max = comb_logit;
#pragma unroll
        for (int mask = 1; mask < HC; mask <<= 1)
          row_max = sycl::fmax(row_max, sycl::permute_group_by_xor(sg, row_max, mask));

        float comb_val = sycl::native::exp2((comb_logit - row_max) * LOG2E);
        float row_sum = comb_val;
#pragma unroll
        for (int mask = 1; mask < HC; mask <<= 1)
          row_sum += sycl::permute_group_by_xor(sg, row_sum, mask);

        comb_val = comb_val * sycl::native::recip(row_sum) + hc_sinkhorn_eps;

        float col_sum = comb_val;
#pragma unroll
        for (int mask = HC; mask < HC2; mask <<= 1)
          col_sum += sycl::permute_group_by_xor(sg, col_sum, mask);
        comb_val = comb_val * sycl::native::recip(col_sum + hc_sinkhorn_eps);

#pragma unroll
        for (int iter = 1; iter < sinkhorn_iters; ++iter) {
          row_sum = comb_val;
#pragma unroll
          for (int mask = 1; mask < HC; mask <<= 1)
            row_sum += sycl::permute_group_by_xor(sg, row_sum, mask);
          comb_val = comb_val * sycl::native::recip(row_sum + hc_sinkhorn_eps);

          col_sum = comb_val;
#pragma unroll
          for (int mask = HC; mask < HC2; mask <<= 1)
            col_sum += sycl::permute_group_by_xor(sg, col_sum, mask);
          comb_val = comb_val * sycl::native::recip(col_sum + hc_sinkhorn_eps);
        }

        comb_mix[static_cast<int64_t>(token_id) * HC2 + lane_id] = comb_val;
      }

    } else {
      // Threads 32-127 (Subgroups 2-7)

      // pre_mix from mixes[:4]
      if (tid >= 32 && tid < 32 + HC) {
        const int pre_idx = tid - 32;
        const float pre_logit = mixes_shared[pre_idx] * hc_scale[0] + hc_base[pre_idx];
        pre_mix_shared[pre_idx] = sycl::native::recip(1.0f + sycl::native::exp2(-pre_logit * LOG2E)) + hc_pre_eps;
      }

      item.barrier(sycl::access::fence_space::local_space);

      // Weighted sum: layer_input[t, h] = sum_k(pre_mix[k] * residual[t, k, h])
      const int threads_for_wsum = WG_SIZE - 32;  // 96 threads
      const int thread_local_id = tid - 32;       // 0..95

      // Stride across hidden_size with all threads
      for (int h = thread_local_id; h < hidden_size; h += threads_for_wsum) {
        float accum = 0.0f;

        for (int k = 0; k < HC; k++) {
          const int64_t res_idx = (static_cast<int64_t>(token_id) * HC + k) * hidden_size + h;
          const float res_val = static_cast<float>(residual[res_idx]);
          accum += pre_mix_shared[k] * res_val;
        }

        layer_input[static_cast<int64_t>(token_id) * hidden_size + h] = static_cast<scalar_t>(accum);
      }
    }
  }
};

void hc_pre_big_fuse(
    const at::Tensor& gemm_out_mul,
    const at::Tensor& gemm_out_sqrsum,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    const at::Tensor& residual_flat,
    at::Tensor& post_mix,
    at::Tensor& comb_mix,
    at::Tensor& layer_input,
    int64_t hc_mult,
    int64_t sinkhorn_iters,
    int64_t n_splits,
    double rms_eps,
    double hc_pre_eps,
    double hc_sinkhorn_eps,
    double hc_post_mult_value) {
  CHECK_INPUT(gemm_out_mul);
  CHECK_INPUT(gemm_out_sqrsum);
  CHECK_INPUT(hc_scale);
  CHECK_INPUT(hc_base);
  CHECK_INPUT(residual_flat);
  CHECK_INPUT(post_mix);
  CHECK_INPUT(comb_mix);
  CHECK_INPUT(layer_input);

  TORCH_CHECK(gemm_out_mul.scalar_type() == at::kFloat, "gemm_out_mul must be float32");
  TORCH_CHECK(gemm_out_sqrsum.scalar_type() == at::kFloat, "gemm_out_sqrsum must be float32");
  TORCH_CHECK(hc_scale.scalar_type() == at::kFloat, "hc_scale must be float32");
  TORCH_CHECK(hc_base.scalar_type() == at::kFloat, "hc_base must be float32");
  TORCH_CHECK(residual_flat.scalar_type() == at::kBFloat16, "residual_flat must be bfloat16");
  TORCH_CHECK(post_mix.scalar_type() == at::kFloat, "post_mix must be float32");
  TORCH_CHECK(comb_mix.scalar_type() == at::kFloat, "comb_mix must be float32");
  TORCH_CHECK(layer_input.scalar_type() == at::kBFloat16, "layer_input must be bfloat16");

  TORCH_CHECK(static_cast<int>(hc_mult) == HC, "hc_mult must be ", HC, ", got ", hc_mult);
  TORCH_CHECK(hc_scale.numel() == 3, "hc_scale must have 3 elements");
  TORCH_CHECK(hc_base.numel() == HC3, "hc_base must have ", HC3, " elements");

  const int64_t n_splits_actual = gemm_out_mul.size(0);
  const int64_t T = gemm_out_mul.size(1);
  TORCH_CHECK(gemm_out_mul.size(2) == HC3, "gemm_out_mul last dim must be ", HC3);
  TORCH_CHECK(gemm_out_sqrsum.size(0) == n_splits_actual, "sqrsum n_splits mismatch");
  TORCH_CHECK(gemm_out_sqrsum.size(1) == T, "sqrsum T mismatch");
  TORCH_CHECK(
      n_splits == n_splits_actual, "n_splits argument (", n_splits, ") must match tensor size (", n_splits_actual, ")");

  TORCH_CHECK(residual_flat.size(0) == T, "residual_flat T mismatch");
  TORCH_CHECK(residual_flat.size(1) == HC, "residual_flat must have ", HC, " channels");
  const int64_t hidden_size = residual_flat.size(2);

  TORCH_CHECK(post_mix.numel() == T * HC, "post_mix size mismatch");
  TORCH_CHECK(comb_mix.numel() == T * HC2, "comb_mix size mismatch");
  TORCH_CHECK(layer_input.numel() == T * hidden_size, "layer_input size mismatch");

  TORCH_CHECK(T < std::numeric_limits<int>::max(), "T too large");
  TORCH_CHECK(hidden_size < std::numeric_limits<int>::max(), "hidden_size too large");

  auto q = dpcppGetCurrentQueue();

  constexpr int slm_size = HC3 + HC;

  using scalar_t = sycl::ext::oneapi::bfloat16;

  q.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> slm(sycl::range<1>(slm_size), cgh);

    auto ker = HCPreBigFuseKernel<scalar_t>{
        gemm_out_mul.data_ptr<float>(),
        gemm_out_sqrsum.data_ptr<float>(),
        hc_scale.data_ptr<float>(),
        hc_base.data_ptr<float>(),
        reinterpret_cast<const scalar_t*>(residual_flat.data_ptr<at::BFloat16>()),
        post_mix.data_ptr<float>(),
        comb_mix.data_ptr<float>(),
        reinterpret_cast<scalar_t*>(layer_input.data_ptr<at::BFloat16>()),
        static_cast<int>(T),
        static_cast<int>(hidden_size),
        static_cast<int>(n_splits),
        static_cast<float>(rms_eps),
        static_cast<float>(hc_pre_eps),
        static_cast<float>(hc_sinkhorn_eps),
        static_cast<float>(hc_post_mult_value),
        static_cast<int>(sinkhorn_iters),
        slm,
    };

    cgh.parallel_for(sycl::nd_range<1>(T * WG_SIZE, WG_SIZE), ker);
  });
}
