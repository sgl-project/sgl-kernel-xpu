#include <ATen/ATen.h>
#include <torch/all.h>

#include <limits>
#include <sycl/sycl.hpp>

#include "SYCLHelpers.h"
#include "Utils.h"

static constexpr int WG_SIZE = 96;  // 6 subgroups of 16 threads each
static constexpr float LOG2E = 1.442695040888963f;

template <typename scalar_t, int VEC_SIZE, int HC = 4>
struct HCPreBigFuseKernelBase : public __SYCL_KER_CONFIG_CONVENTION__ {
  static constexpr int HC2 = HC * HC;
  static constexpr int HC3 = (2 + HC) * HC;
  static constexpr int SINKHORN_ITERS = 20;
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

  sycl::local_accessor<float, 1> slm_;

  HCPreBigFuseKernelBase(
      const float* gemm_out_mul_,
      const float* gemm_out_sqrsum_,
      const float* hc_scale_,
      const float* hc_base_,
      const scalar_t* residual_,
      float* post_mix_,
      float* comb_mix_,
      scalar_t* layer_input_,
      int T_total_,
      int hidden_size_,
      int n_splits_,
      float rms_eps_,
      float hc_pre_eps_,
      float hc_sinkhorn_eps_,
      float hc_post_mult_value_)
      : gemm_out_mul(gemm_out_mul_),
        gemm_out_sqrsum(gemm_out_sqrsum_),
        hc_scale(hc_scale_),
        hc_base(hc_base_),
        residual(residual_),
        post_mix(post_mix_),
        comb_mix(comb_mix_),
        layer_input(layer_input_),
        T_total(T_total_),
        hidden_size(hidden_size_),
        n_splits(n_splits_),
        rms_eps(rms_eps_),
        hc_pre_eps(hc_pre_eps_),
        hc_sinkhorn_eps(hc_sinkhorn_eps_),
        hc_post_mult_value(hc_post_mult_value_) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    constexpr int slm_size = HC3 + HC;
    slm_ = sycl::local_accessor<float, 1>(slm_size, cgh);
  }

  // RMS normalization and Sinkhorn computation
  inline float* compute_rms_and_sinkhorn(
      sycl::nd_item<1> item, sycl::sub_group sg, int token_id, int tid, int sg_id, int lane_id) const {
    float* mixes_shared = slm_.get_multi_ptr<sycl::access::decorated::no>().get();
    float* pre_mix_shared = mixes_shared + HC3;

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

    // mix operations
    if (sg_id == 0) {
      // pre_mix, post_mix computed by lanes 0-3
      if (lane_id < HC) {
        // pre_mix from mixes[0:4]
        const float pre_logit = mixes_shared[lane_id] * hc_scale[0] + hc_base[lane_id];
        pre_mix_shared[lane_id] = sycl::native::recip(1.0f + sycl::native::exp2(-pre_logit * LOG2E)) + hc_pre_eps;

        // post_mix from mixes[4:8]
        const float post_logit = mixes_shared[HC + lane_id] * hc_scale[1] + hc_base[HC + lane_id];
        post_mix[static_cast<int64_t>(token_id) * HC + lane_id] =
            hc_post_mult_value * sycl::native::recip(1.0f + sycl::native::exp2(-post_logit * LOG2E));
      }

      // comb_mix (Sinkhorn) computed by all 16 lanes from mixes[8:24]
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
      for (int iter = 1; iter < SINKHORN_ITERS; ++iter) {
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

    item.barrier(sycl::access::fence_space::local_space);

    return pre_mix_shared;
  }
};

template <typename scalar_t, int VEC_SIZE, int HC = 4>
struct HCPreBigFuseKernel : public HCPreBigFuseKernelBase<scalar_t, VEC_SIZE, HC> {
  using Base = HCPreBigFuseKernelBase<scalar_t, VEC_SIZE, HC>;

  HCPreBigFuseKernel(
      const float* gemm_out_mul_,
      const float* gemm_out_sqrsum_,
      const float* hc_scale_,
      const float* hc_base_,
      const scalar_t* residual_,
      float* post_mix_,
      float* comb_mix_,
      scalar_t* layer_input_,
      int T_total_,
      int hidden_size_,
      int n_splits_,
      float rms_eps_,
      float hc_pre_eps_,
      float hc_sinkhorn_eps_,
      float hc_post_mult_value_)
      : Base(
            gemm_out_mul_,
            gemm_out_sqrsum_,
            hc_scale_,
            hc_base_,
            residual_,
            post_mix_,
            comb_mix_,
            layer_input_,
            T_total_,
            hidden_size_,
            n_splits_,
            rms_eps_,
            hc_pre_eps_,
            hc_sinkhorn_eps_,
            hc_post_mult_value_) {}

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    sycl::sub_group sg = item.get_sub_group();

    const int token_id = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    const int sg_id = static_cast<int>(sg.get_group_id()[0]);
    const int lane_id = static_cast<int>(sg.get_local_id()[0]);

    if (token_id >= this->T_total) return;

    float* pre_mix_shared = this->compute_rms_and_sinkhorn(item, sg, token_id, tid, sg_id, lane_id);

    // Weighted sum
    const int threads_for_wsum = WG_SIZE;
    const int thread_local_id = tid;

    constexpr int kVecSize = VEC_SIZE;
    using vec_in = vec_t<scalar_t, kVecSize>;
    using vec_out = vec_t<scalar_t, kVecSize>;

    const int num_vec_elems = this->hidden_size / kVecSize;
    const int vec_tail_start = num_vec_elems * kVecSize;

    for (int i = thread_local_id; i < num_vec_elems; i += threads_for_wsum) {
      const int h = i * kVecSize;

      float accum[kVecSize] = {};

      for (int k = 0; k < HC; k++) {
        const int64_t res_base = (static_cast<int64_t>(token_id) * HC + k) * this->hidden_size + h;
        vec_in v;
        v.load(
            0, sycl::multi_ptr<const scalar_t, sycl::access::address_space::global_space>(&this->residual[res_base]));

#pragma unroll
        for (int j = 0; j < kVecSize; ++j) {
          accum[j] += pre_mix_shared[k] * static_cast<float>(v[j]);
        }
      }

      vec_out out;
#pragma unroll
      for (int j = 0; j < kVecSize; ++j) {
        out[j] = static_cast<scalar_t>(accum[j]);
      }
      out.store(
          0,
          sycl::multi_ptr<scalar_t, sycl::access::address_space::global_space>(
              &this->layer_input[static_cast<int64_t>(token_id) * this->hidden_size + h]));
    }

    for (int h = vec_tail_start + thread_local_id; h < this->hidden_size; h += threads_for_wsum) {
      float accum = 0.0f;
      for (int k = 0; k < HC; k++) {
        const int64_t res_idx = (static_cast<int64_t>(token_id) * HC + k) * this->hidden_size + h;
        accum += pre_mix_shared[k] * static_cast<float>(this->residual[res_idx]);
      }
      this->layer_input[static_cast<int64_t>(token_id) * this->hidden_size + h] = static_cast<scalar_t>(accum);
    }
  }
};

template <typename scalar_t, int VEC_SIZE, int HC = 4>
struct HCPreBigFuseWithNormKernel : public HCPreBigFuseKernelBase<scalar_t, VEC_SIZE, HC> {
  using Base = HCPreBigFuseKernelBase<scalar_t, VEC_SIZE, HC>;

  const scalar_t* __restrict__ norm_weight;
  float norm_eps;

  HCPreBigFuseWithNormKernel(
      const float* gemm_out_mul_,
      const float* gemm_out_sqrsum_,
      const float* hc_scale_,
      const float* hc_base_,
      const scalar_t* residual_,
      float* post_mix_,
      float* comb_mix_,
      scalar_t* layer_input_,
      const scalar_t* norm_weight_,
      int T_total_,
      int hidden_size_,
      int n_splits_,
      float rms_eps_,
      float hc_pre_eps_,
      float hc_sinkhorn_eps_,
      float hc_post_mult_value_,
      float norm_eps_)
      : Base(
            gemm_out_mul_,
            gemm_out_sqrsum_,
            hc_scale_,
            hc_base_,
            residual_,
            post_mix_,
            comb_mix_,
            layer_input_,
            T_total_,
            hidden_size_,
            n_splits_,
            rms_eps_,
            hc_pre_eps_,
            hc_sinkhorn_eps_,
            hc_post_mult_value_),
        norm_weight(norm_weight_),
        norm_eps(norm_eps_) {}

  [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> item) const {
    sycl::sub_group sg = item.get_sub_group();

    const int token_id = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    const int sg_id = static_cast<int>(sg.get_group_id()[0]);
    const int lane_id = static_cast<int>(sg.get_local_id()[0]);

    if (token_id >= this->T_total) return;

    float* pre_mix_shared = this->compute_rms_and_sinkhorn(item, sg, token_id, tid, sg_id, lane_id);

    // Weighted sum with local square-sum calculation
    if (tid >= 16) {
      const int thread_local_id = tid - 16;
      const int stride = 80;

      constexpr int kVecSize = VEC_SIZE;
      using vec_in = vec_t<scalar_t, kVecSize>;

      const int num_vec_elems = this->hidden_size / kVecSize;
      const int vec_tail_start = num_vec_elems * kVecSize;

      float local_sqr_sum = 0.0f;
      for (int i = thread_local_id; i < num_vec_elems; i += stride) {
        const int h = i * kVecSize;
        float accum[kVecSize] = {};

        for (int k = 0; k < HC; k++) {
          const int64_t res_base = (static_cast<int64_t>(token_id) * HC + k) * this->hidden_size + h;
          vec_in v;
          v.load(
              0, sycl::multi_ptr<const scalar_t, sycl::access::address_space::global_space>(&this->residual[res_base]));

#pragma unroll
          for (int j = 0; j < kVecSize; ++j) {
            accum[j] += pre_mix_shared[k] * static_cast<float>(v[j]);
          }
        }

#pragma unroll
        for (int j = 0; j < kVecSize; ++j) {
          local_sqr_sum += accum[j] * accum[j];
        }
      }

      for (int h = vec_tail_start + thread_local_id; h < this->hidden_size; h += stride) {
        float accum = 0.0f;
        for (int k = 0; k < HC; k++) {
          const int64_t res_idx = (static_cast<int64_t>(token_id) * HC + k) * this->hidden_size + h;
          accum += pre_mix_shared[k] * static_cast<float>(this->residual[res_idx]);
        }
        local_sqr_sum += accum * accum;
      }

      local_sqr_sum = sycl::reduce_over_group(sg, local_sqr_sum, sycl::plus<float>());
      if (lane_id == 0) {
        this->slm_[sg_id - 1] = local_sqr_sum;
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    if (sg_id == 0) {
      const int idx = sycl::select(0, lane_id, lane_id < 5);
      const float val_to_reduce = sycl::select(0.0f, this->slm_[idx], lane_id < 5);
      const float total_sqr_sum = sycl::reduce_over_group(sg, val_to_reduce, sycl::plus<float>());
      if (lane_id == 0) {
        const float mean_sqr = total_sqr_sum / static_cast<float>(this->hidden_size);
        this->slm_[0] = sycl::rsqrt(mean_sqr + norm_eps);
      }
    }

    item.barrier(sycl::access::fence_space::local_space);

    // recompute with RMS normalization
    if (tid >= 16) {
      const int thread_local_id = tid - 16;
      const int stride = 80;
      const float rms = this->slm_[0];

      constexpr int kVecSize = VEC_SIZE;
      using vec_in = vec_t<scalar_t, kVecSize>;
      using vec_out = vec_t<scalar_t, kVecSize>;

      const int num_vec_elems = this->hidden_size / kVecSize;
      const int vec_tail_start = num_vec_elems * kVecSize;

      for (int i = thread_local_id; i < num_vec_elems; i += stride) {
        const int h = i * kVecSize;
        float accum[kVecSize] = {};

        for (int k = 0; k < HC; k++) {
          const int64_t res_base = (static_cast<int64_t>(token_id) * HC + k) * this->hidden_size + h;
          vec_in v;
          v.load(
              0, sycl::multi_ptr<const scalar_t, sycl::access::address_space::global_space>(&this->residual[res_base]));

#pragma unroll
          for (int j = 0; j < kVecSize; ++j) {
            accum[j] += pre_mix_shared[k] * static_cast<float>(v[j]);
          }
        }

        vec_in weight_vec;
        weight_vec.load(0, sycl::multi_ptr<const scalar_t, sycl::access::address_space::global_space>(&norm_weight[h]));

        vec_out out;
#pragma unroll
        for (int j = 0; j < kVecSize; ++j) {
          out[j] = static_cast<scalar_t>(accum[j] * rms * static_cast<float>(weight_vec[j]));
        }
        out.store(
            0,
            sycl::multi_ptr<scalar_t, sycl::access::address_space::global_space>(
                &this->layer_input[static_cast<int64_t>(token_id) * this->hidden_size + h]));
      }

      for (int h = vec_tail_start + thread_local_id; h < this->hidden_size; h += stride) {
        float accum = 0.0f;
        for (int k = 0; k < HC; k++) {
          const int64_t res_idx = (static_cast<int64_t>(token_id) * HC + k) * this->hidden_size + h;
          accum += pre_mix_shared[k] * static_cast<float>(this->residual[res_idx]);
        }
        const float weight = static_cast<float>(norm_weight[h]);
        this->layer_input[static_cast<int64_t>(token_id) * this->hidden_size + h] =
            static_cast<scalar_t>(accum * rms * weight);
      }
    }
  }
};

template <int VEC_SIZE>
static void launch_hc_pre_fuse_kernel(
    sycl::queue& q,
    const at::Tensor& gemm_out_mul,
    const at::Tensor& gemm_out_sqrsum,
    const at::Tensor& hc_scale,
    const at::Tensor& hc_base,
    const at::Tensor& residual_flat,
    at::Tensor& post_mix,
    at::Tensor& comb_mix,
    at::Tensor& layer_input,
    int64_t T,
    int64_t hidden_size,
    int64_t n_splits,
    double rms_eps,
    double hc_pre_eps,
    double hc_sinkhorn_eps,
    double hc_post_mult_value,
    std::optional<at::Tensor> norm_weight,
    std::optional<double> norm_eps) {
  using scalar_t = sycl::ext::oneapi::bfloat16;

  if (norm_weight.has_value()) {
    const float norm_eps_val = static_cast<float>(norm_eps.value_or(1e-6));
    auto ker = HCPreBigFuseWithNormKernel<scalar_t, VEC_SIZE>(
        gemm_out_mul.data_ptr<float>(),
        gemm_out_sqrsum.data_ptr<float>(),
        hc_scale.data_ptr<float>(),
        hc_base.data_ptr<float>(),
        reinterpret_cast<const scalar_t*>(residual_flat.data_ptr<at::BFloat16>()),
        post_mix.data_ptr<float>(),
        comb_mix.data_ptr<float>(),
        reinterpret_cast<scalar_t*>(layer_input.data_ptr<at::BFloat16>()),
        reinterpret_cast<const scalar_t*>(norm_weight.value().data_ptr<at::BFloat16>()),
        static_cast<int>(T),
        static_cast<int>(hidden_size),
        static_cast<int>(n_splits),
        static_cast<float>(rms_eps),
        static_cast<float>(hc_pre_eps),
        static_cast<float>(hc_sinkhorn_eps),
        static_cast<float>(hc_post_mult_value),
        norm_eps_val);
    sycl_kernel_submit(sycl::range<1>(T * WG_SIZE), sycl::range<1>(WG_SIZE), q, ker);
  } else {
    auto ker = HCPreBigFuseKernel<scalar_t, VEC_SIZE>(
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
        static_cast<float>(hc_post_mult_value));
    sycl_kernel_submit(sycl::range<1>(T * WG_SIZE), sycl::range<1>(WG_SIZE), q, ker);
  }
}

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
    double hc_post_mult_value,
    std::optional<at::Tensor> norm_weight,
    std::optional<double> norm_eps) {
  CHECK_INPUT(gemm_out_mul);
  CHECK_INPUT(gemm_out_sqrsum);
  CHECK_INPUT(hc_scale);
  CHECK_INPUT(hc_base);
  CHECK_INPUT(residual_flat);
  CHECK_INPUT(post_mix);
  CHECK_INPUT(comb_mix);
  CHECK_INPUT(layer_input);

  if (norm_weight.has_value()) {
    CHECK_INPUT(norm_weight.value());
    TORCH_CHECK(norm_weight.value().scalar_type() == at::kBFloat16, "norm_weight must be bfloat16");
  }

  TORCH_CHECK(gemm_out_mul.scalar_type() == at::kFloat, "gemm_out_mul must be float32");
  TORCH_CHECK(gemm_out_sqrsum.scalar_type() == at::kFloat, "gemm_out_sqrsum must be float32");
  TORCH_CHECK(hc_scale.scalar_type() == at::kFloat, "hc_scale must be float32");
  TORCH_CHECK(hc_base.scalar_type() == at::kFloat, "hc_base must be float32");
  TORCH_CHECK(residual_flat.scalar_type() == at::kBFloat16, "residual_flat must be bfloat16");
  TORCH_CHECK(post_mix.scalar_type() == at::kFloat, "post_mix must be float32");
  TORCH_CHECK(comb_mix.scalar_type() == at::kFloat, "comb_mix must be float32");
  TORCH_CHECK(layer_input.scalar_type() == at::kBFloat16, "layer_input must be bfloat16");

  constexpr int HC = 4;
  constexpr int HC3 = (2 + HC) * HC;
  constexpr int HC2 = HC * HC;
  constexpr int SINKHORN_ITERS = 20;

  TORCH_CHECK(static_cast<int>(hc_mult) == HC, "hc_mult must be ", HC, ", got ", hc_mult);
  TORCH_CHECK(
      static_cast<int>(sinkhorn_iters) == SINKHORN_ITERS,
      "sinkhorn_iters must be ",
      SINKHORN_ITERS,
      ", got ",
      sinkhorn_iters);
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

  if (norm_weight.has_value()) {
    TORCH_CHECK(norm_weight.value().numel() == hidden_size, "norm_weight size must match hidden_size");
  }

  auto q = dpcppGetCurrentQueue();

  // Dynamic VEC_SIZE selection based on batch size
  int vec_size = (T <= 16) ? 8 : (T <= 48) ? 4 : 2;

#define LAUNCH_HC_PRE_FUSE(VEC_SIZE)     \
  case VEC_SIZE: {                       \
    launch_hc_pre_fuse_kernel<VEC_SIZE>( \
        q,                               \
        gemm_out_mul,                    \
        gemm_out_sqrsum,                 \
        hc_scale,                        \
        hc_base,                         \
        residual_flat,                   \
        post_mix,                        \
        comb_mix,                        \
        layer_input,                     \
        T,                               \
        hidden_size,                     \
        n_splits,                        \
        rms_eps,                         \
        hc_pre_eps,                      \
        hc_sinkhorn_eps,                 \
        hc_post_mult_value,              \
        norm_weight,                     \
        norm_eps);                       \
    break;                               \
  }

  switch (vec_size) {
    LAUNCH_HC_PRE_FUSE(8);
    LAUNCH_HC_PRE_FUSE(4);
    LAUNCH_HC_PRE_FUSE(2);
    default:
      TORCH_CHECK(false, "Unsupported vec_size: ", vec_size);
  }

#undef LAUNCH_HC_PRE_FUSE
}
