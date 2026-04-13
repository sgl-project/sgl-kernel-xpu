#include <ATen/ATen.h>
#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sycl/sycl.hpp>
#include <tuple>

#include "SYCLHelpers.h"
#include "Utils.h"

namespace {

constexpr int kWgSize = 64;
constexpr int kMaxHeadDim = 256;
constexpr int kUpdateWgSize = 256;
constexpr int kUpdateSubGroupSize = 32;
constexpr int kChunkSeqBlock = 64;

#define LAUNCH_BUCKET_KERNEL(launch_fn, bucket_size) launch_fn(std::integral_constant<int, bucket_size>{})

#define LAUNCH_BUCKET_KERNEL_WITH_L2_TAG(launch_fn, l2_tag, bucket_size) \
  launch_fn(l2_tag, std::integral_constant<int, bucket_size>{})

template <typename scalar_t>
inline float to_float(scalar_t v) {
  return static_cast<float>(v);
}

template <typename scalar_t>
inline scalar_t from_float(float v) {
  return static_cast<scalar_t>(v);
}

inline float sigmoidf(float x) {
  return 1.0f / (1.0f + sycl::exp(-x));
}

inline float softplus_with_threshold(float x, float beta, float threshold) {
  const float bx = beta * x;
  if (bx > threshold) {
    return x;
  }
  return sycl::log1p(sycl::exp(bx)) / beta;
}

template <typename scalar_t>
struct FusedGdnGatingKernel {
  const float* A_log;
  const scalar_t* a;
  const scalar_t* b;
  const scalar_t* dt_bias;
  float* g_out;
  scalar_t* beta_out;
  int64_t batch;
  int64_t num_heads;
  float beta;
  float threshold;

  void operator()(sycl::nd_item<1> item) const {
    const int64_t idx = item.get_global_linear_id();
    const int64_t total = batch * num_heads;
    if (idx >= total) {
      return;
    }
    const int64_t b_idx = idx / num_heads;
    const int64_t h = idx % num_heads;
    const float x = to_float(a[idx]) + to_float(dt_bias[h]);
    const float g = -sycl::exp(A_log[h]) * softplus_with_threshold(x, beta, threshold);
    const float beta_val = sigmoidf(to_float(b[idx]));

    g_out[idx] = g;
    beta_out[idx] = from_float<scalar_t>(beta_val);
  }
};

template <typename scalar_t, int kBucketSize>
struct ChunkGatedDeltaRuleKernel {
  const scalar_t* q;
  const scalar_t* k;
  const scalar_t* v;
  const float* g;
  const scalar_t* beta;
  const int32_t* cu_seqlens;
  const int32_t* chunk_indices;
  int32_t* seq_head_progress;
  scalar_t* out;
  float* final_state;
  int64_t num_chunks;
  int64_t chunk_size;
  int64_t num_seqs;
  int64_t q_heads;
  int64_t v_heads;
  int64_t head_dim;
  int64_t value_head_dim;
  int64_t head_expand;
  int64_t q_stride_b;
  int64_t q_stride_t;
  int64_t q_stride_h;
  int64_t q_stride_d;
  int64_t k_stride_b;
  int64_t k_stride_t;
  int64_t k_stride_h;
  int64_t k_stride_d;
  int64_t v_stride_b;
  int64_t v_stride_t;
  int64_t v_stride_h;
  int64_t v_stride_d;
  int64_t out_stride_b;
  int64_t out_stride_t;
  int64_t out_stride_h;
  int64_t out_stride_d;
  int use_qk_l2norm;
  float eps;
  float q_scale;

  void operator()(sycl::nd_item<2> item) const {
    const int64_t chunk_gid = item.get_global_id(0);
    if (chunk_gid >= num_chunks) {
      return;
    }
    const int64_t vh = item.get_global_id(1);
    if (vh >= v_heads) {
      return;
    }

    const int64_t seq_idx = static_cast<int64_t>(chunk_indices[chunk_gid * 2]);
    const int64_t local_chunk_idx = static_cast<int64_t>(chunk_indices[chunk_gid * 2 + 1]);
    if (seq_idx < 0 || seq_idx >= num_seqs || local_chunk_idx < 0) {
      return;
    }

    const int64_t qh = vh / head_expand;
    const int64_t seq_start = static_cast<int64_t>(cu_seqlens[seq_idx]);
    const int64_t seq_end = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
    const int64_t start = seq_start + local_chunk_idx * chunk_size;
    if (start >= seq_end) {
      return;
    }
    const int64_t end = std::min(start + chunk_size, seq_end);

    const int64_t progress_idx = seq_idx * v_heads + vh;
    sycl::global_ptr<int32_t> progress_ptr(seq_head_progress + progress_idx);
    sycl::atomic_ref<
        int32_t,
        sycl::memory_order::acq_rel,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space>
        progress_atomic(*progress_ptr);

    while (progress_atomic.load() != static_cast<int32_t>(local_chunk_idx)) {
    }

    float* state = final_state + ((seq_idx * v_heads + vh) * head_dim * value_head_dim);
    float kv_mem[kMaxHeadDim];
    float delta[kMaxHeadDim];
    scalar_t q_cache[kMaxHeadDim];
    scalar_t k_cache[kMaxHeadDim];

    for (int64_t t = start; t < end; ++t) {
      float q_norm_sq = 0.0f;
      float k_norm_sq = 0.0f;
      if (use_qk_l2norm) {
        int64_t d = 0;
        for (; d + kBucketSize - 1 < head_dim; d += kBucketSize) {
          const int64_t q_base = q_stride_b * 0 + t * q_stride_t + qh * q_stride_h + d * q_stride_d;
          const int64_t k_base = k_stride_b * 0 + t * k_stride_t + qh * k_stride_h + d * k_stride_d;
#pragma unroll
          for (int i = 0; i < kBucketSize; ++i) {
            const scalar_t qv = q[q_base + i * q_stride_d];
            const scalar_t kv = k[k_base + i * k_stride_d];
            q_norm_sq += static_cast<float>(qv * qv);
            k_norm_sq += static_cast<float>(kv * kv);
          }
        }
        for (; d < head_dim; ++d) {
          const int64_t q_off = q_stride_b * 0 + t * q_stride_t + qh * q_stride_h + d * q_stride_d;
          const int64_t k_off = k_stride_b * 0 + t * k_stride_t + qh * k_stride_h + d * k_stride_d;
          const scalar_t qv = q[q_off];
          const scalar_t kv = k[k_off];
          q_norm_sq += static_cast<float>(qv * qv);
          k_norm_sq += static_cast<float>(kv * kv);
        }
      }
      const float q_inv = use_qk_l2norm ? sycl::rsqrt(q_norm_sq + eps) : 1.0f;
      const float k_inv = use_qk_l2norm ? sycl::rsqrt(k_norm_sq + eps) : 1.0f;
      const scalar_t q_scale_t = from_float<scalar_t>(q_inv * q_scale);
      const scalar_t k_scale_t = from_float<scalar_t>(k_inv);

      int64_t d = 0;
      for (; d + kBucketSize - 1 < head_dim; d += kBucketSize) {
        const int64_t q_base = q_stride_b * 0 + t * q_stride_t + qh * q_stride_h + d * q_stride_d;
        const int64_t k_base = k_stride_b * 0 + t * k_stride_t + qh * k_stride_h + d * k_stride_d;
#pragma unroll
        for (int i = 0; i < kBucketSize; ++i) {
          q_cache[d + i] = q[q_base + i * q_stride_d] * q_scale_t;
          k_cache[d + i] = k[k_base + i * k_stride_d] * k_scale_t;
        }
      }
      for (; d < head_dim; ++d) {
        const int64_t q_off = q_stride_b * 0 + t * q_stride_t + qh * q_stride_h + d * q_stride_d;
        const int64_t k_off = k_stride_b * 0 + t * k_stride_t + qh * k_stride_h + d * k_stride_d;
        q_cache[d] = q[q_off] * q_scale_t;
        k_cache[d] = k[k_off] * k_scale_t;
      }

      const int64_t gb_off = (t * v_heads + vh);
      const float g_t = sycl::exp(g[gb_off]);
      const scalar_t beta_t = beta[gb_off];

      const int64_t state_elems = head_dim * value_head_dim;
      int64_t s = 0;
      for (; s + 3 < state_elems; s += 4) {
        state[s] *= g_t;
        state[s + 1] *= g_t;
        state[s + 2] *= g_t;
        state[s + 3] *= g_t;
      }
      for (; s < state_elems; ++s) {
        state[s] *= g_t;
      }

      constexpr int kVec = 8;
      int64_t dv = 0;
      for (; dv + kVec - 1 < value_head_dim; dv += kVec) {
        sycl::vec<float, kVec> kv_vec(0.0f);
        for (int64_t dk = 0; dk < head_dim; ++dk) {
          const float k_val = static_cast<float>(k_cache[dk]);
          const sycl::vec<float, kVec> k_vec(k_val);
          const float* s_ptr = state + dk * value_head_dim + dv;
          sycl::vec<float, kVec> s_vec(0.0f);
#pragma unroll
          for (int i = 0; i < kVec; ++i) {
            s_vec[i] = s_ptr[i];
          }
          kv_vec += s_vec * k_vec;
        }
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          const int64_t dv_i = dv + i;
          const int64_t v_off = v_stride_b * 0 + t * v_stride_t + vh * v_stride_h + dv_i * v_stride_d;
          const float kv_val = kv_vec[i];
          const scalar_t kv_cast = from_float<scalar_t>(kv_val);
          kv_mem[dv_i] = kv_val;
          delta[dv_i] = static_cast<float>((v[v_off] - kv_cast) * beta_t);
        }
      }
      for (; dv < value_head_dim; ++dv) {
        float acc = 0.0f;
        for (int64_t dk = 0; dk < head_dim; ++dk) {
          acc += state[dk * value_head_dim + dv] * static_cast<float>(k_cache[dk]);
        }
        const int64_t v_off = v_stride_b * 0 + t * v_stride_t + vh * v_stride_h + dv * v_stride_d;
        const scalar_t kv_cast = from_float<scalar_t>(acc);
        kv_mem[dv] = acc;
        delta[dv] = static_cast<float>((v[v_off] - kv_cast) * beta_t);
      }

      for (int64_t dk = 0; dk < head_dim; ++dk) {
        const float k_val = static_cast<float>(k_cache[dk]);
        const sycl::vec<float, kVec> k_vec(k_val);
        float* s_ptr = state + dk * value_head_dim;
        int64_t dv_vec = 0;
        for (; dv_vec + kVec - 1 < value_head_dim; dv_vec += kVec) {
          sycl::vec<float, kVec> s_vec(0.0f);
          sycl::vec<float, kVec> d_vec(0.0f);
#pragma unroll
          for (int i = 0; i < kVec; ++i) {
            s_vec[i] = s_ptr[dv_vec + i];
            d_vec[i] = delta[dv_vec + i];
          }
          s_vec += d_vec * k_vec;
#pragma unroll
          for (int i = 0; i < kVec; ++i) {
            s_ptr[dv_vec + i] = s_vec[i];
          }
        }
        for (; dv_vec < value_head_dim; ++dv_vec) {
          s_ptr[dv_vec] += k_val * delta[dv_vec];
        }
      }

      dv = 0;
      for (; dv + kVec - 1 < value_head_dim; dv += kVec) {
        sycl::vec<float, kVec> out_vec(0.0f);
        for (int64_t dk = 0; dk < head_dim; ++dk) {
          const float q_val = static_cast<float>(q_cache[dk]);
          const sycl::vec<float, kVec> q_vec(q_val);
          const float* s_ptr = state + dk * value_head_dim + dv;
          sycl::vec<float, kVec> s_vec(0.0f);
#pragma unroll
          for (int i = 0; i < kVec; ++i) {
            s_vec[i] = s_ptr[i];
          }
          out_vec += s_vec * q_vec;
        }
#pragma unroll
        for (int i = 0; i < kVec; ++i) {
          const int64_t dv_i = dv + i;
          const int64_t o_off = out_stride_b * 0 + t * out_stride_t + vh * out_stride_h + dv_i * out_stride_d;
          out[o_off] = from_float<scalar_t>(out_vec[i]);
        }
      }
      for (; dv < value_head_dim; ++dv) {
        float o = 0.0f;
        for (int64_t dk = 0; dk < head_dim; ++dk) {
          o += state[dk * value_head_dim + dv] * static_cast<float>(q_cache[dk]);
        }
        const int64_t o_off = out_stride_b * 0 + t * out_stride_t + vh * out_stride_h + dv * out_stride_d;
        out[o_off] = from_float<scalar_t>(o);
      }
    }

    progress_atomic.store(static_cast<int32_t>(local_chunk_idx + 1));
  }
};

template <typename scalar_t, int kBucketSize, bool kUseQkL2Norm>
struct FusedSigmoidGatingDeltaRuleUpdateKernel {
  static constexpr int sub_group_size = kUpdateSubGroupSize;
  static constexpr int group_size = kUpdateWgSize;
  static constexpr int sg_per_group = group_size / sub_group_size;
  static constexpr int v_dim_per_sg = 4;
  static constexpr int v_dim_per_group = v_dim_per_sg * sg_per_group;

  const float* A_log;
  const scalar_t* dt_bias;
  const scalar_t* q;
  const scalar_t* k;
  const scalar_t* v;
  const scalar_t* a;
  const scalar_t* b;
  const int32_t* initial_state_indices;
  scalar_t* out;
  float* initial_state_source;
  const int32_t* cu_seqlens;
  int64_t batch;
  int64_t q_heads;
  int64_t v_heads;
  int64_t head_dim;
  int64_t value_head_dim;
  int64_t head_expand;
  int64_t q_stride_b;
  int64_t q_stride_t;
  int64_t q_stride_h;
  int64_t q_stride_d;
  int64_t k_stride_b;
  int64_t k_stride_t;
  int64_t k_stride_h;
  int64_t k_stride_d;
  int64_t v_stride_b;
  int64_t v_stride_t;
  int64_t v_stride_h;
  int64_t v_stride_d;
  float eps;
  float q_scale;
  float softplus_beta;
  float softplus_threshold;

  static inline sycl::nd_range<3> get_nd_range(int64_t batch_size, int64_t num_v_heads, int64_t head_v_dim) {
    const int64_t num_v_bucket = (head_v_dim + v_dim_per_group - 1) / v_dim_per_group;
    sycl::range<3> local(1, 1, group_size);
    sycl::range<3> global(batch_size, num_v_heads, num_v_bucket);
    return sycl::nd_range<3>(global * local, local);
  }

  [[intel::reqd_sub_group_size(sub_group_size)]] void operator()(sycl::nd_item<3> item) const {
    const int64_t b_idx = item.get_group(0);
    const int64_t vh = item.get_group(1);
    const int64_t v_bucket = item.get_group(2);

    auto sg = item.get_sub_group();
    const int sg_id = sg.get_group_linear_id();
    const int sg_local_id = sg.get_local_linear_id();

    const int64_t dv_base = v_bucket * v_dim_per_group + sg_id * v_dim_per_sg;
    if (dv_base >= value_head_dim) {
      return;
    }

    const int64_t qh = vh / head_expand;
    const int64_t state_index = static_cast<int64_t>(initial_state_indices[b_idx]);
    float* state = initial_state_source + ((state_index * v_heads + vh) * head_dim * value_head_dim);
    float state_local[v_dim_per_sg * kBucketSize];
    float q_local[kBucketSize];
    float k_local[kBucketSize];
    float delta_local[v_dim_per_sg];

    const float x = to_float(a[b_idx * v_heads + vh]) + to_float(dt_bias[vh]);
    const float g_base = -sycl::exp(A_log[vh]) * softplus_with_threshold(x, softplus_beta, softplus_threshold);
    const float g_t = sycl::exp(g_base);
    const float beta_t = sigmoidf(to_float(b[b_idx * v_heads + vh]));

#pragma unroll
    for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
      for (int i = 0; i < kBucketSize; ++i) {
        const int64_t dk = kBucketSize * sg_local_id + i;
        state_local[j * kBucketSize + i] = state[dk * value_head_dim + dv_base + j];
      }
    }

    const int64_t seq_start = static_cast<int64_t>(cu_seqlens[b_idx]);
    const int64_t seq_end = static_cast<int64_t>(cu_seqlens[b_idx + 1]);
    for (int64_t t = seq_start; t < seq_end; ++t) {
      const int64_t local_t = t - seq_start;
      float q_norm_sq = 0.0f;
      float k_norm_sq = 0.0f;
      if constexpr (kUseQkL2Norm) {
#pragma unroll
        for (int i = 0; i < kBucketSize; ++i) {
          const int64_t dk = kBucketSize * sg_local_id + i;
          const int64_t q_off = b_idx * q_stride_b + local_t * q_stride_t + qh * q_stride_h + dk * q_stride_d;
          const int64_t k_off = b_idx * k_stride_b + local_t * k_stride_t + qh * k_stride_h + dk * k_stride_d;
          q_local[i] = to_float(q[q_off]);
          k_local[i] = to_float(k[k_off]);
          q_norm_sq += q_local[i] * q_local[i];
          k_norm_sq += k_local[i] * k_local[i];
        }
        q_norm_sq = sycl::reduce_over_group(sg, q_norm_sq, sycl::plus<>());
        k_norm_sq = sycl::reduce_over_group(sg, k_norm_sq, sycl::plus<>());
      } else {
#pragma unroll
        for (int i = 0; i < kBucketSize; ++i) {
          const int64_t dk = kBucketSize * sg_local_id + i;
          const int64_t q_off = b_idx * q_stride_b + local_t * q_stride_t + qh * q_stride_h + dk * q_stride_d;
          const int64_t k_off = b_idx * k_stride_b + local_t * k_stride_t + qh * k_stride_h + dk * k_stride_d;
          q_local[i] = to_float(q[q_off]);
          k_local[i] = to_float(k[k_off]);
        }
      }
      const float q_inv = kUseQkL2Norm ? sycl::rsqrt(q_norm_sq + eps) : 1.0f;
      const float k_inv = kUseQkL2Norm ? sycl::rsqrt(k_norm_sq + eps) : 1.0f;

#pragma unroll
      for (int i = 0; i < kBucketSize; ++i) {
        q_local[i] = q_local[i] * q_inv * q_scale;
        k_local[i] = k_local[i] * k_inv;
      }

      float kv_mem[v_dim_per_sg];
#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
        kv_mem[j] = 0.0f;
      }

#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < kBucketSize; ++i) {
          state_local[j * kBucketSize + i] *= g_t;
          kv_mem[j] += state_local[j * kBucketSize + i] * k_local[i];
        }
      }

#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
        kv_mem[j] = sycl::reduce_over_group(sg, kv_mem[j], sycl::plus<>());
        const int64_t v_off = b_idx * v_stride_b + local_t * v_stride_t + vh * v_stride_h + (dv_base + j) * v_stride_d;
        delta_local[j] = (to_float(v[v_off]) - kv_mem[j]) * beta_t;
      }

      float out_local[v_dim_per_sg];
#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
        out_local[j] = 0.0f;
      }

#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < kBucketSize; ++i) {
          state_local[j * kBucketSize + i] += k_local[i] * delta_local[j];
          out_local[j] += state_local[j * kBucketSize + i] * q_local[i];
        }
      }

#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
        out_local[j] = sycl::reduce_over_group(sg, out_local[j], sycl::plus<>());
      }

      if (sg_local_id == 0) {
#pragma unroll
        for (int j = 0; j < v_dim_per_sg; ++j) {
          const int64_t o_off =
              (((b_idx * (seq_end - seq_start) + (t - seq_start)) * v_heads + vh) * value_head_dim + dv_base + j);
          out[o_off] = from_float<scalar_t>(out_local[j]);
        }
      }
    }

#pragma unroll
    for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
      for (int i = 0; i < kBucketSize; ++i) {
        const int64_t dk = kBucketSize * sg_local_id + i;
        state[dk * value_head_dim + dv_base + j] = state_local[j * kBucketSize + i];
      }
    }
  }
};

}  // namespace

std::tuple<at::Tensor, at::Tensor> chunk_gated_delta_rule(
    const at::Tensor& query,
    const at::Tensor& key,
    const at::Tensor& value,
    const at::Tensor& g,
    const at::Tensor& beta,
    const at::Tensor& initial_state,
    bool output_final_state,
    const at::Tensor& cu_seqlens,
    const at::Tensor& chunk_indices,
    const at::Tensor& chunk_offsets,
    bool head_first,
    bool use_qk_l2norm_in_kernel,
    double eps) {
  TORCH_CHECK(!head_first, "chunk_gated_delta_rule does not support head_first=True");
  CHECK_DIM(4, query);
  CHECK_DIM(4, key);
  CHECK_DIM(4, value);
  CHECK_DIM(3, g);
  CHECK_DIM(3, beta);
  CHECK_DIM(4, initial_state);
  CHECK_DIM(1, cu_seqlens);
  CHECK_DIM(2, chunk_indices);
  CHECK_DIM(1, chunk_offsets);
  CHECK_EQ(query.size(0), 1);
  CHECK_DEVICE(query);
  CHECK_DEVICE(key);
  CHECK_DEVICE(value);
  CHECK_INPUT(g);
  CHECK_INPUT(beta);
  CHECK_INPUT(initial_state);
  CHECK_INPUT(cu_seqlens);
  CHECK_INPUT(chunk_indices);
  CHECK_INPUT(chunk_offsets);

  const int64_t q_heads = query.size(2);
  const int64_t head_dim = query.size(3);
  const int64_t v_heads = value.size(2);
  const int64_t value_head_dim = value.size(3);
  const int64_t num_seqs = initial_state.size(0);

  TORCH_CHECK(v_heads % q_heads == 0, "expect value heads to be a multiple of query heads");
  TORCH_CHECK(head_dim <= kMaxHeadDim, "head_dim too large for kernel temporary buffers");
  TORCH_CHECK(value_head_dim <= kMaxHeadDim, "value_head_dim too large for kernel temporary buffers");
  TORCH_CHECK(initial_state.scalar_type() == at::kFloat, "initial_state must be float32");
  TORCH_CHECK(g.scalar_type() == at::kFloat, "g must be float32");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt, "cu_seqlens must be int32");
  TORCH_CHECK(chunk_indices.scalar_type() == at::kInt, "chunk_indices must be int32");
  TORCH_CHECK(chunk_offsets.scalar_type() == at::kInt, "chunk_offsets must be int32");
  TORCH_CHECK(cu_seqlens.size(0) == num_seqs + 1, "cu_seqlens size must be num_seqs + 1");
  TORCH_CHECK(chunk_offsets.size(0) == num_seqs + 1, "chunk_offsets size must be num_seqs + 1");
  TORCH_CHECK(chunk_indices.size(1) == 2, "chunk_indices must have shape [num_chunks, 2]");

  at::Tensor output = at::empty_like(value, value.options());
  at::Tensor final_state = initial_state.contiguous().clone();
  at::Tensor seq_head_progress = at::zeros({num_seqs, v_heads}, chunk_indices.options());

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  const int64_t num_chunks = chunk_indices.size(0);
  const int64_t global_heads = ((v_heads + kWgSize - 1) / kWgSize) * kWgSize;
  sycl::range<2> local_range(1, kWgSize);
  sycl::range<2> global_range(num_chunks, global_heads);
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t head_expand = v_heads / q_heads;
  const int64_t q_stride_b = query.stride(0);
  const int64_t q_stride_t = query.stride(1);
  const int64_t q_stride_h = query.stride(2);
  const int64_t q_stride_d = query.stride(3);
  const int64_t k_stride_b = key.stride(0);
  const int64_t k_stride_t = key.stride(1);
  const int64_t k_stride_h = key.stride(2);
  const int64_t k_stride_d = key.stride(3);
  const int64_t v_stride_b = value.stride(0);
  const int64_t v_stride_t = value.stride(1);
  const int64_t v_stride_h = value.stride(2);
  const int64_t v_stride_d = value.stride(3);
  const int64_t out_stride_b = output.stride(0);
  const int64_t out_stride_t = output.stride(1);
  const int64_t out_stride_h = output.stride(2);
  const int64_t out_stride_d = output.stride(3);

  SYCL_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, query.scalar_type(), "chunk_gated_delta_rule", [&]() {
        auto launch_kernel = [&](auto bucket_size_tag) {
          constexpr int kBucket = decltype(bucket_size_tag)::value;
          ChunkGatedDeltaRuleKernel<scalar_t, kBucket> kfn{
              .q = reinterpret_cast<scalar_t*>(query.data_ptr()),
              .k = reinterpret_cast<scalar_t*>(key.data_ptr()),
              .v = reinterpret_cast<scalar_t*>(value.data_ptr()),
              .g = g.data_ptr<float>(),
              .beta = reinterpret_cast<scalar_t*>(beta.data_ptr()),
              .cu_seqlens = cu_seqlens.data_ptr<int32_t>(),
              .chunk_indices = chunk_indices.data_ptr<int32_t>(),
              .seq_head_progress = seq_head_progress.data_ptr<int32_t>(),
              .out = reinterpret_cast<scalar_t*>(output.data_ptr()),
              .final_state = final_state.data_ptr<float>(),
              .num_chunks = num_chunks,
              .chunk_size = kChunkSeqBlock,
              .num_seqs = num_seqs,
              .q_heads = q_heads,
              .v_heads = v_heads,
              .head_dim = head_dim,
              .value_head_dim = value_head_dim,
              .head_expand = head_expand,
              .q_stride_b = q_stride_b,
              .q_stride_t = q_stride_t,
              .q_stride_h = q_stride_h,
              .q_stride_d = q_stride_d,
              .k_stride_b = k_stride_b,
              .k_stride_t = k_stride_t,
              .k_stride_h = k_stride_h,
              .k_stride_d = k_stride_d,
              .v_stride_b = v_stride_b,
              .v_stride_t = v_stride_t,
              .v_stride_h = v_stride_h,
              .v_stride_d = v_stride_d,
              .out_stride_b = out_stride_b,
              .out_stride_t = out_stride_t,
              .out_stride_h = out_stride_h,
              .out_stride_d = out_stride_d,
              .use_qk_l2norm = use_qk_l2norm_in_kernel ? 1 : 0,
              .eps = static_cast<float>(eps),
              .q_scale = q_scale,
          };
          sycl_kernel_submit(global_range, local_range, queue, kfn);
        };

        switch (head_dim) {
          case 64:
            LAUNCH_BUCKET_KERNEL(launch_kernel, 2);
            break;
          case 128:
            LAUNCH_BUCKET_KERNEL(launch_kernel, 4);
            break;
          case 256:
            LAUNCH_BUCKET_KERNEL(launch_kernel, 8);
            break;
          default:
            LAUNCH_BUCKET_KERNEL(launch_kernel, 1);
            break;
        }
      });

  if (!output_final_state) {
    return std::make_tuple(output, at::Tensor());
  }
  return std::make_tuple(output, final_state);
}

std::tuple<at::Tensor, at::Tensor>
fused_gdn_gating(const at::Tensor& A_log, const at::Tensor& a, const at::Tensor& b, const at::Tensor& dt_bias) {
  CHECK_DIM(1, A_log);
  CHECK_DIM(2, a);
  CHECK_DIM(2, b);
  CHECK_DIM(1, dt_bias);
  CHECK_EQ(A_log.size(0), a.size(1));
  CHECK_EQ(A_log.size(0), b.size(1));
  CHECK_EQ(A_log.size(0), dt_bias.size(0));
  CHECK_INPUT(A_log);
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(dt_bias);
  TORCH_CHECK(A_log.scalar_type() == at::kFloat, "A_log must be float32");

  const int64_t batch = a.size(0);
  const int64_t num_heads = a.size(1);

  at::Tensor g = at::empty({1, batch, num_heads}, a.options().dtype(at::kFloat));
  at::Tensor beta_out = at::empty({1, batch, num_heads}, b.options());

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();
  const int64_t total_items = batch * num_heads;
  const int64_t global = ((total_items + kWgSize - 1) / kWgSize) * kWgSize;

  DISPATCH_FLOAT_TYPES(a.scalar_type(), "fused_gdn_gating", [&] {
    FusedGdnGatingKernel<scalar_t> kfn{
        .A_log = A_log.data_ptr<float>(),
        .a = a.data_ptr<scalar_t>(),
        .b = b.data_ptr<scalar_t>(),
        .dt_bias = dt_bias.data_ptr<scalar_t>(),
        .g_out = g.data_ptr<float>(),
        .beta_out = beta_out.data_ptr<scalar_t>(),
        .batch = batch,
        .num_heads = num_heads,
        .beta = 1.0f,
        .threshold = 20.0f,
    };
    sycl_kernel_submit(global, kWgSize, queue, kfn);
  });

  return std::make_tuple(g, beta_out);
}

at::Tensor fused_sigmoid_gating_delta_rule_update(
    const at::Tensor& A_log,
    const at::Tensor& dt_bias,
    const at::Tensor& q,
    const at::Tensor& k,
    const at::Tensor& v,
    const at::Tensor& a,
    const at::Tensor& b,
    at::Tensor& initial_state_source,
    const at::Tensor& initial_state_indices,
    const at::Tensor& cu_seqlens,
    bool use_qk_l2norm_in_kernel,
    double softplus_beta,
    double softplus_threshold) {
  CHECK_DIM(4, q);
  CHECK_DIM(4, k);
  CHECK_DIM(4, v);
  CHECK_DIM(1, A_log);
  CHECK_DIM(2, a);
  CHECK_DIM(2, b);
  CHECK_DIM(1, dt_bias);
  CHECK_DIM(1, initial_state_indices);
  CHECK_DIM(1, cu_seqlens);
  CHECK_DIM(4, initial_state_source);
  CHECK_INPUT(A_log);
  CHECK_INPUT(dt_bias);
  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(v);
  CHECK_INPUT(a);
  CHECK_INPUT(b);
  CHECK_INPUT(initial_state_source);
  CHECK_INPUT(initial_state_indices);
  CHECK_INPUT(cu_seqlens);

  const int64_t seq_len = q.size(0);
  TORCH_CHECK(
      seq_len == 1, "fused_sigmoid_gating_delta_rule_update only used in decode where sequence length must be 1");
  const int64_t batch_size = q.size(1);
  const int64_t q_heads = q.size(2);
  const int64_t head_dim = q.size(3);
  const int64_t v_heads = v.size(2);
  const int64_t value_head_dim = v.size(3);
  TORCH_CHECK(v_heads % q_heads == 0, "expect value heads to be a multiple of query heads");
  TORCH_CHECK(head_dim <= kMaxHeadDim, "head_dim too large for kernel temporary buffers");
  TORCH_CHECK(value_head_dim <= kMaxHeadDim, "value_head_dim too large for kernel temporary buffers");
  TORCH_CHECK(head_dim % kUpdateSubGroupSize == 0, "head_dim must be divisible by subgroup size");
  TORCH_CHECK(
      value_head_dim % (4 * (kUpdateWgSize / kUpdateSubGroupSize)) == 0,
      "value_head_dim must be divisible by workgroup output tile size");
  TORCH_CHECK(A_log.scalar_type() == at::kFloat, "A_log must be float32");
  TORCH_CHECK(initial_state_source.scalar_type() == at::kFloat, "initial_state_source must be float32");
  TORCH_CHECK(initial_state_indices.scalar_type() == at::kInt, "initial_state_indices must be int32");

  at::Tensor out = at::empty({batch_size, seq_len, v_heads, value_head_dim}, q.options());

  auto stream = at::xpu::getCurrentXPUStream();
  auto queue = stream.queue();

  const float q_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t head_expand = v_heads / q_heads;
  const int64_t q_stride_t = q.stride(0);
  const int64_t q_stride_b = q.stride(1);
  const int64_t q_stride_h = q.stride(2);
  const int64_t q_stride_d = q.stride(3);
  const int64_t k_stride_t = k.stride(0);
  const int64_t k_stride_b = k.stride(1);
  const int64_t k_stride_h = k.stride(2);
  const int64_t k_stride_d = k.stride(3);
  const int64_t v_stride_t = v.stride(0);
  const int64_t v_stride_b = v.stride(1);
  const int64_t v_stride_h = v.stride(2);
  const int64_t v_stride_d = v.stride(3);

  SYCL_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16, at::ScalarType::Half, q.scalar_type(), "fused_sigmoid_gating_delta_rule_update", [&]() {
        auto launch_kernel = [&](auto use_l2norm_tag, auto bucket_size_tag) {
          constexpr bool kUseL2Norm = decltype(use_l2norm_tag)::value;
          constexpr int kBucketSize = decltype(bucket_size_tag)::value;
          using Kernel = FusedSigmoidGatingDeltaRuleUpdateKernel<scalar_t, kBucketSize, kUseL2Norm>;
          Kernel kfn{
              .A_log = A_log.data_ptr<float>(),
              .dt_bias = reinterpret_cast<scalar_t*>(dt_bias.data_ptr()),
              .q = reinterpret_cast<scalar_t*>(q.data_ptr()),
              .k = reinterpret_cast<scalar_t*>(k.data_ptr()),
              .v = reinterpret_cast<scalar_t*>(v.data_ptr()),
              .a = reinterpret_cast<scalar_t*>(a.data_ptr()),
              .b = reinterpret_cast<scalar_t*>(b.data_ptr()),
              .initial_state_indices = initial_state_indices.data_ptr<int32_t>(),
              .out = reinterpret_cast<scalar_t*>(out.data_ptr()),
              .initial_state_source = initial_state_source.data_ptr<float>(),
              .cu_seqlens = cu_seqlens.data_ptr<int32_t>(),
              .batch = batch_size,
              .q_heads = q_heads,
              .v_heads = v_heads,
              .head_dim = head_dim,
              .value_head_dim = value_head_dim,
              .head_expand = head_expand,
              .q_stride_b = q_stride_b,
              .q_stride_t = q_stride_t,
              .q_stride_h = q_stride_h,
              .q_stride_d = q_stride_d,
              .k_stride_b = k_stride_b,
              .k_stride_t = k_stride_t,
              .k_stride_h = k_stride_h,
              .k_stride_d = k_stride_d,
              .v_stride_b = v_stride_b,
              .v_stride_t = v_stride_t,
              .v_stride_h = v_stride_h,
              .v_stride_d = v_stride_d,
              .eps = 1e-6f,
              .q_scale = q_scale,
              .softplus_beta = static_cast<float>(softplus_beta),
              .softplus_threshold = static_cast<float>(softplus_threshold),
          };
          auto nd_range = Kernel::get_nd_range(batch_size, v_heads, value_head_dim);
          sycl_kernel_submit(nd_range.get_global_range(), nd_range.get_local_range(), queue, kfn);
        };

        if (use_qk_l2norm_in_kernel) {
          switch (head_dim) {
            case 64:
              LAUNCH_BUCKET_KERNEL_WITH_L2_TAG(launch_kernel, std::true_type{}, 2);
              break;
            case 128:
              LAUNCH_BUCKET_KERNEL_WITH_L2_TAG(launch_kernel, std::true_type{}, 4);
              break;
            case 256:
              LAUNCH_BUCKET_KERNEL_WITH_L2_TAG(launch_kernel, std::true_type{}, 8);
              break;
            default:
              TORCH_CHECK(false, "unsupported head_dim for subgroup kernel: ", head_dim);
          }
        } else {
          switch (head_dim) {
            case 64:
              LAUNCH_BUCKET_KERNEL_WITH_L2_TAG(launch_kernel, std::false_type{}, 2);
              break;
            case 128:
              LAUNCH_BUCKET_KERNEL_WITH_L2_TAG(launch_kernel, std::false_type{}, 4);
              break;
            case 256:
              LAUNCH_BUCKET_KERNEL_WITH_L2_TAG(launch_kernel, std::false_type{}, 8);
              break;
            default:
              TORCH_CHECK(false, "unsupported head_dim for subgroup kernel: ", head_dim);
          }
        }
      });

  return out;
}

#undef LAUNCH_BUCKET_KERNEL
#undef LAUNCH_BUCKET_KERNEL_WITH_L2_TAG
