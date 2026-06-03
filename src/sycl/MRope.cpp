#include <c10/xpu/XPUStream.h>
#include <torch/all.h>

#include <optional>
#include <sycl/sycl.hpp>
#include <vector>

#include "SYCLHelpers.h"
#include "Utils.h"
#include "comm/Numerics.h"

namespace at::native::xpu {

namespace {

static constexpr int sg_size = 32;

// Adapted from triton implementation in sglang "_triton_mrope_forward_fused":
// https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/rotary_embedding/triton_kernels.py

template <typename scalar_t, bool is_neox, bool is_interleaved>
struct MRoPEKernel {
  // Grid  : nd_range<2>( {num_tokens, global_rotary_dim}, {1, local_rotary_dim} )
  //   dim 0: token   (1 work-group row per token)
  //   dim 1: rot lane (local_rotary_dim lanes per work-group)

  [[sycl::reqd_sub_group_size(sg_size)]]
  void operator()(sycl::nd_item<2> item) const {
    const int32_t token_id = item.get_global_id(0);
    const int32_t rot = item.get_global_id(1);

    if (token_id >= num_tokens_ || rot >= half_rotary_dim_) return;

    const int64_t t_pos = positions_[token_id];
    const int64_t h_pos = positions_[positions_stride_ + token_id];
    const int64_t w_pos = positions_[2 * positions_stride_ + token_id];

    bool h_mask = false, w_mask = false;

    if constexpr (is_interleaved) {
      // Interleaved layout: t=0 mod3, h=1 mod3, w=2 mod3
      h_mask = ((rot % 3) == 1) && (rot < 3 * section_h_);
      w_mask = ((rot % 3) == 2) && (rot < 3 * section_w_);
    } else {
      // Standard contiguous layout: [t_section | h_section | w_section]
      const int64_t t_end = section_t_;
      const int64_t h_end = t_end + section_h_;
      h_mask = (t_end <= rot) && (rot < h_end);
      w_mask = (h_end <= rot) && (rot < half_rotary_dim_);
    }

    const int64_t chosen_pos = h_mask ? h_pos : (w_mask ? w_pos : t_pos);
    const int64_t cache_base = chosen_pos * rotary_dim_;

    const scalar_t cos_v = cos_sin_cache_[cache_base + rot];
    const scalar_t sin_v = cos_sin_cache_[cache_base + rot + half_rotary_dim_];

    const int64_t max_heads = (num_q_heads_ > num_k_heads_) ? num_q_heads_ : num_k_heads_;

    for (int head_id = 0; head_id < max_heads; ++head_id) {
      if (head_id < num_q_heads_) {
        scalar_t* q_head = query_ + token_id * q_stride_ + head_id * head_size_;

        if constexpr (is_neox) {
          const scalar_t x = q_head[rot];
          const scalar_t y = q_head[rot + half_rotary_dim_];
          q_head[rot] = x * cos_v - y * sin_v;
          q_head[rot + half_rotary_dim_] = y * cos_v + x * sin_v;
        } else {
          const int64_t x_idx = 2 * rot;
          const int64_t y_idx = x_idx + 1;
          const scalar_t x = q_head[x_idx];
          const scalar_t y = q_head[y_idx];
          q_head[x_idx] = x * cos_v - y * sin_v;
          q_head[y_idx] = y * cos_v + x * sin_v;
        }
      }
      if (head_id < num_k_heads_) {
        scalar_t* k_head = key_ + token_id * k_stride_ + head_id * head_size_;

        if constexpr (is_neox) {
          const scalar_t x = k_head[rot];
          const scalar_t y = k_head[rot + half_rotary_dim_];
          k_head[rot] = x * cos_v - y * sin_v;
          k_head[rot + half_rotary_dim_] = y * cos_v + x * sin_v;
        } else {
          const int64_t x_idx = 2 * rot;
          const int64_t y_idx = x_idx + 1;
          const scalar_t x = k_head[x_idx];
          const scalar_t y = k_head[y_idx];
          k_head[x_idx] = x * cos_v - y * sin_v;
          k_head[y_idx] = y * cos_v + x * sin_v;
        }
      }
    }
  }

  // Pointers
  scalar_t* query_;
  scalar_t* key_;
  const scalar_t* cos_sin_cache_;
  const int64_t* positions_;

  // Strides
  int64_t q_stride_;
  int64_t k_stride_;
  int64_t positions_stride_;

  // Sizes
  int64_t num_tokens_;
  int64_t num_q_heads_;
  int64_t num_k_heads_;
  int64_t head_size_;
  int64_t rotary_dim_;
  int64_t half_rotary_dim_;  // rd/2

  // MRoPE section boundaries (in units of rot-lane index)
  int64_t section_t_;
  int64_t section_h_;
  int64_t section_w_;
};

template <typename scalar_t, bool is_neox, bool is_interleaved>
void launch_mrope(
    at::Tensor& query,
    at::Tensor& key,
    const at::Tensor& cos_sin_cache,
    const at::Tensor& positions,
    int64_t num_q_heads,
    int64_t num_k_heads,
    int64_t head_size,
    int64_t rotary_dim,
    int64_t section_t,
    int64_t section_h,
    int64_t section_w) {
  const int64_t num_tokens = query.size(0);
  const int64_t half_rotary_dim = rotary_dim / 2;

  // local_rotary_dim(wg_size): sub-group-friendly tile size for the rotary dimension.
  const int64_t local_rotary_dim = RoundUp(half_rotary_dim, static_cast<int64_t>(sg_size));
  const int64_t global_rotary_dim = RoundUp(half_rotary_dim, local_rotary_dim);

  MRoPEKernel<scalar_t, is_neox, is_interleaved> kernel{
      query.data_ptr<scalar_t>(),
      key.data_ptr<scalar_t>(),
      cos_sin_cache.data_ptr<scalar_t>(),
      positions.data_ptr<int64_t>(),
      static_cast<int64_t>(query.stride(0)),
      static_cast<int64_t>(key.stride(0)),
      static_cast<int64_t>(positions.stride(0)),
      num_tokens,
      num_q_heads,
      num_k_heads,
      head_size,
      rotary_dim,
      half_rotary_dim,
      section_t,
      section_h,
      section_w,
  };

  sycl_kernel_submit(
      sycl::range<2>(num_tokens, global_rotary_dim),
      sycl::range<2>(1, local_rotary_dim),
      dpcppGetCurrentQueue(),
      kernel);
}

}  // namespace

void multimodal_rotary_embedding(
    at::Tensor& query,
    at::Tensor& key,
    at::Tensor& cos_sin_cache,
    at::Tensor& positions,
    const std::vector<int64_t>& mrope_section,
    int64_t head_size,
    int64_t rotary_dim,
    bool mrope_interleaved,
    bool mrope_interleaved_glm,
    bool is_neox_style,
    const std::optional<at::Tensor>& axis_map) {
  TORCH_CHECK(query.dim() == 2, "query must be 2D [num_tokens, num_heads * head_size]");
  TORCH_CHECK(key.dim() == 2, "key must be 2D [num_tokens, num_kv_heads * head_size]");
  TORCH_CHECK(mrope_section.size() == 3, "mrope_section must have 3 elements [t, h, w]");

  const int64_t section_t = mrope_section[0];
  const int64_t section_h = mrope_section[1];
  const int64_t section_w = mrope_section[2];
  TORCH_CHECK(section_t >= 0 && section_h >= 0 && section_w >= 0, "mrope_section values must be non-negative");
  if (!mrope_interleaved) {
    TORCH_CHECK(
        section_t + section_h + section_w == rotary_dim / 2,
        "for non-interleaved mrope, sum(mrope_section) must equal rotary_dim/2");
  }

  const int64_t num_tokens = query.size(0);
  TORCH_CHECK(positions.size(1) == num_tokens, "positions second dimension must equal num_tokens");
  TORCH_CHECK(mrope_interleaved_glm == false, "glm style rope is not supported");

  const int64_t num_q_heads = query.size(1) / head_size;
  const int64_t num_k_heads = key.size(1) / head_size;

#define LAUNCH_MROPE_KERNEL(IS_NEOX, IS_INTERLEAVED)                                                                 \
  SYCL_DISPATCH_FLOATING_TYPES(at::kHalf, at::kBFloat16, query.scalar_type(), "multimodal_rotary_embedding", [&]() { \
    launch_mrope<scalar_t, IS_NEOX, IS_INTERLEAVED>(                                                                 \
        query,                                                                                                       \
        key,                                                                                                         \
        cos_sin_cache,                                                                                               \
        positions,                                                                                                   \
        num_q_heads,                                                                                                 \
        num_k_heads,                                                                                                 \
        head_size,                                                                                                   \
        rotary_dim,                                                                                                  \
        section_t,                                                                                                   \
        section_h,                                                                                                   \
        section_w);                                                                                                  \
  });

  if (is_neox_style) {
    if (mrope_interleaved) {
      LAUNCH_MROPE_KERNEL(true, true);
    } else {
      LAUNCH_MROPE_KERNEL(true, false);
    }
  } else {
    if (mrope_interleaved) {
      LAUNCH_MROPE_KERNEL(false, true);
    } else {
      LAUNCH_MROPE_KERNEL(false, false);
    }
  }
#undef LAUNCH_MROPE_KERNEL
}

}  // namespace at::native::xpu
