#include <ATen/ATen.h>
#include <ATen/OpMathType.h>
#include <c10/util/Float8_e4m3fn.h>
#include <torch/all.h>

#include <cmath>
#include <cstdint>
#include <sycl/sycl.hpp>

#include "QKNormCommon.h"
#include "Utils.h"
#include "cutlass/float8.h"

using cutlass::float_e4m3_t;

namespace at::native::xpu {

static inline float
compute_freq_yarn_rope(float log2_base, int rotary_dim, int half_dim, float factor, float low, float high) {
  const float exponent = -2.0f * static_cast<float>(half_dim) / static_cast<float>(rotary_dim);
  float freq = sycl::exp2(exponent * log2_base);

  if (factor != 1.0f) {
    float inv_freq_extrapolation = freq;
    float inv_freq_interpolation = freq / factor;

    float high_adj = high;
    if (sycl::fabs(low - high_adj) <= 1e-6f) {
      high_adj += 0.001f;
    }

    float dim_value = 2.0f * static_cast<float>(half_dim);
    float linear_func = (dim_value - low) / (high_adj - low);
    float ramp_func = sycl::fmin(sycl::fmax(linear_func, 0.0f), 1.0f);
    freq = inv_freq_interpolation * (1.0f - ramp_func) + inv_freq_extrapolation * ramp_func;
  }

  return freq;
}

template <bool interleave>
struct QKNormRopePostOp {
  const int* position_ids;
  float log2_base;
  float factor;
  float low;
  float high;
  float attention_factor;
  int rotary_dim;

  int cta_staging_elems(int head_dim, int vec_size) const {
    return std::min(head_dim, div_up(rotary_dim, vec_size) * vec_size);
  }

  template <typename accscalar_t, int num_tiles, int vec_size, int head_dim>
  static accscalar_t
  load_warp_dim(accscalar_t (&elements)[num_tiles][vec_size], sycl::sub_group sg, int lane_id, int dim) {
    constexpr int tile_width = NUM_REDUCE_STAGES * vec_size;
    const int tile = dim / tile_width;
    const int tile_offset = dim - tile * tile_width;
    const int target_lane = tile_offset / vec_size;
    const int value_index = tile_offset - target_lane * vec_size;

    const accscalar_t value = elements[tile][value_index];
    // Always call permute_group_by_xor for SYCL spec compliance (all lanes must participate).
    // When target_lane == lane_id the XOR mask is 0, so each lane receives its own value.
    return sycl::permute_group_by_xor(sg, value, lane_id ^ target_lane);
  }

  template <typename accscalar_t, int num_tiles, int vec_size, int head_dim>
  void apply_warp_full_interleave(accscalar_t (&elements)[num_tiles][vec_size], int lane_id, float pos_id) const {
    const accscalar_t scale = static_cast<accscalar_t>(attention_factor);
#pragma unroll
    for (int tile = 0; tile < num_tiles; ++tile) {
#pragma unroll
      for (int v = 0; v < vec_size; v += 2) {
        const int dim = (tile * NUM_REDUCE_STAGES + lane_id) * vec_size + v;
        const float freq = compute_freq_yarn_rope(log2_base, head_dim, dim / 2, factor, low, high);
        const float theta = pos_id * freq;
        const accscalar_t sin_val = static_cast<accscalar_t>(sycl::native::sin(theta));
        const accscalar_t cos_val = static_cast<accscalar_t>(sycl::native::cos(theta));
        // Read both elements before writing either; v and v+1 are a pair processed atomically.
        const accscalar_t x0 = elements[tile][v];
        const accscalar_t x1 = elements[tile][v + 1];
        elements[tile][v] = (x0 * cos_val - x1 * sin_val) * scale;
        elements[tile][v + 1] = (x1 * cos_val + x0 * sin_val) * scale;
      }
    }
  }

  template <typename accscalar_t, int num_tiles, int vec_size, int head_dim>
  void apply_warp_full_neox(
      accscalar_t (&elements)[num_tiles][vec_size], sycl::sub_group sg, int lane_id, float pos_id) const {
    constexpr int half_rotary_dim = head_dim / 2;
    constexpr int tile_width = NUM_REDUCE_STAGES * vec_size;
    constexpr int lane_xor = (half_rotary_dim / vec_size) % NUM_REDUCE_STAGES;
    constexpr int tile_delta = half_rotary_dim / tile_width;
    const accscalar_t scale = static_cast<accscalar_t>(attention_factor);

    if constexpr (half_rotary_dim % tile_width == 0) {
      // Tile-delta path: each pair (dim, dim+half_rotary_dim) lives in two different tiles
      // of the SAME lane.  Process both tiles together so no copy of elements is needed
      // and the shared sin/cos is computed only once per pair.
      // tile_delta == num_tiles / 2 when half_rotary_dim % tile_width == 0.
#pragma unroll
      for (int tile = 0; tile < tile_delta; ++tile) {
        const int paired_tile = tile + tile_delta;
#pragma unroll
        for (int v = 0; v < vec_size; ++v) {
          const int dim = (tile * NUM_REDUCE_STAGES + lane_id) * vec_size + v;
          // dim is in the first half; dim + half_rotary_dim is in the second half.
          const accscalar_t xa = elements[tile][v];
          const accscalar_t xb = elements[paired_tile][v];
          const int freq_dim = (dim * 2) % head_dim;  // same for dim and dim+half_rotary_dim
          const float freq = compute_freq_yarn_rope(log2_base, head_dim, freq_dim / 2, factor, low, high);
          const float theta = pos_id * freq;
          const accscalar_t sin_val = static_cast<accscalar_t>(sycl::native::sin(theta));
          const accscalar_t cos_val = static_cast<accscalar_t>(sycl::native::cos(theta));
          // first half:  x*cos + (-xb)*sin
          // second half: x*cos + xa*sin
          elements[tile][v] = (xa * cos_val - xb * sin_val) * scale;
          elements[paired_tile][v] = (xb * cos_val + xa * sin_val) * scale;
        }
      }
    } else {
      // XOR-shuffle path: paired elements are in different lanes of the SAME tile.
      // The shuffle reads all lanes' values simultaneously before any write, so
      // elements can be used directly without a copy.
#pragma unroll
      for (int tile = 0; tile < num_tiles; ++tile) {
#pragma unroll
        for (int v = 0; v < vec_size; ++v) {
          const int dim = (tile * NUM_REDUCE_STAGES + lane_id) * vec_size + v;
          const bool first_half = dim < half_rotary_dim;
          // Shuffle reads all lanes' values before this lane writes.
          const accscalar_t xb = sycl::permute_group_by_xor(sg, elements[tile][v], lane_xor);
          const accscalar_t rotated = first_half ? -xb : xb;
          const int freq_dim = (dim * 2) % head_dim;
          const float freq = compute_freq_yarn_rope(log2_base, head_dim, freq_dim / 2, factor, low, high);
          const float theta = pos_id * freq;
          const accscalar_t sin_val = static_cast<accscalar_t>(sycl::native::sin(theta));
          const accscalar_t cos_val = static_cast<accscalar_t>(sycl::native::cos(theta));
          const accscalar_t x = elements[tile][v];
          elements[tile][v] = (x * cos_val + rotated * sin_val) * scale;
        }
      }
    }
  }

  template <typename accscalar_t, int num_tiles, int vec_size, int head_dim>
  void apply_warp_generic(
      accscalar_t (&elements)[num_tiles][vec_size], sycl::sub_group sg, int lane_id, float pos_id) const {
    const accscalar_t scale = static_cast<accscalar_t>(attention_factor);
    // A copy is required: for neox partial-rotary the second-half elements read their
    // paired dim from the first half, which may have been written in an earlier tile
    // iteration.  (Interleave and xor-neox fast paths avoid this cost.)
    accscalar_t original[num_tiles][vec_size];
#pragma unroll
    for (int tile = 0; tile < num_tiles; ++tile) {
#pragma unroll
      for (int v = 0; v < vec_size; ++v) {
        original[tile][v] = elements[tile][v];
      }
    }

#pragma unroll
    for (int tile = 0; tile < num_tiles; ++tile) {
#pragma unroll
      for (int v = 0; v < vec_size; ++v) {
        const int dim = (tile * NUM_REDUCE_STAGES + lane_id) * vec_size + v;
        const bool in_rotary = (dim < rotary_dim);
        int paired_dim;
        int half_dim;
        bool negate;

        if constexpr (interleave) {
          // Use dim itself as a dummy paired_dim for out-of-rotary lanes so that
          // load_dim's group shuffle is called unconditionally by all lanes
          // (SYCL spec requires all work-items to participate in group operations).
          paired_dim = in_rotary ? (dim ^ 1) : dim;
          negate = in_rotary && ((dim & 1) == 0);
          half_dim = dim / 2;
        } else {
          const int half_rotary_dim = rotary_dim / 2;
          const bool first_half = dim < half_rotary_dim;
          paired_dim = in_rotary ? (first_half ? dim + half_rotary_dim : dim - half_rotary_dim) : dim;
          negate = in_rotary && first_half;
          const int freq_dim = (dim * 2) % rotary_dim;
          half_dim = freq_dim / 2;
        }

        // Always call load_dim so all lanes participate in the group shuffle.
        accscalar_t rotated =
            load_warp_dim<accscalar_t, num_tiles, vec_size, head_dim>(original, sg, lane_id, paired_dim);

        if (in_rotary) {
          if (negate) {
            rotated = -rotated;
          }
          const accscalar_t x = original[tile][v];
          const float freq = compute_freq_yarn_rope(log2_base, rotary_dim, half_dim, factor, low, high);
          const float theta = pos_id * freq;
          const accscalar_t sin_val = static_cast<accscalar_t>(sycl::native::sin(theta));
          const accscalar_t cos_val = static_cast<accscalar_t>(sycl::native::cos(theta));
          elements[tile][v] = (x * cos_val + rotated * sin_val) * scale;
        }
      }
    }
  }

  template <typename accscalar_t, int num_tiles, int vec_size, int head_dim>
  void apply_warp_path(
      accscalar_t (&elements)[num_tiles][vec_size],
      sycl::nd_item<1> item,
      int64_t token_id,
      int64_t,
      bool,
      int64_t lane_id_value,
      int64_t) const {
    auto sg = item.get_sub_group();
    const int lane_id = static_cast<int>(lane_id_value);
    const float pos_id = static_cast<float>(position_ids[token_id]);

    if (rotary_dim == head_dim) {
      if constexpr (interleave && vec_size % 2 == 0) {
        apply_warp_full_interleave<accscalar_t, num_tiles, vec_size, head_dim>(elements, lane_id, pos_id);
      } else if constexpr (!interleave) {
        apply_warp_full_neox<accscalar_t, num_tiles, vec_size, head_dim>(elements, sg, lane_id, pos_id);
      } else {
        apply_warp_generic<accscalar_t, num_tiles, vec_size, head_dim>(elements, sg, lane_id, pos_id);
      }
    } else {
      apply_warp_generic<accscalar_t, num_tiles, vec_size, head_dim>(elements, sg, lane_id, pos_id);
    }
  }

  // CTA path: the normalized head lives in shared local memory, so paired RoPE
  // dimensions are read directly from `head` and work for any head size.
  template <typename accscalar_t, typename HeadT>
  accscalar_t apply_cta_path(const HeadT& head, int dim, int64_t token_id) const {
    const accscalar_t x = head[dim];
    if (dim >= rotary_dim) {
      return x;
    }
    const float pos_id = static_cast<float>(position_ids[token_id]);
    int paired_dim;
    int half_dim;
    bool negate;
    if constexpr (interleave) {
      paired_dim = dim ^ 1;
      negate = ((dim & 1) == 0);
      half_dim = dim / 2;
    } else {
      const int half_rotary_dim = rotary_dim / 2;
      const bool first_half = dim < half_rotary_dim;
      paired_dim = first_half ? dim + half_rotary_dim : dim - half_rotary_dim;
      negate = first_half;
      const int freq_dim = (dim * 2) % rotary_dim;
      half_dim = freq_dim / 2;
    }
    accscalar_t rotated = head[paired_dim];
    if (negate) {
      rotated = -rotated;
    }
    const float freq = compute_freq_yarn_rope(log2_base, rotary_dim, half_dim, factor, low, high);
    const float theta = pos_id * freq;
    const accscalar_t sin_val = static_cast<accscalar_t>(sycl::native::sin(theta));
    const accscalar_t cos_val = static_cast<accscalar_t>(sycl::native::cos(theta));
    return (x * cos_val + rotated * sin_val) * static_cast<accscalar_t>(attention_factor);
  }
};

static void validate_rotary_for_vec_size(int64_t head_dim, int64_t rotary_dim, bool is_neox, int vec_size) {
  TORCH_CHECK(
      rotary_dim > 0 && rotary_dim <= head_dim, "rotary_dim must be in the range (0, head_dim], got ", rotary_dim);
  TORCH_CHECK(rotary_dim % 2 == 0, "rotary_dim must be even for RoPE, got ", rotary_dim);

  if (is_neox) {
    const int64_t half_rotary_dim = rotary_dim / 2;
    if (half_rotary_dim > vec_size) {
      TORCH_CHECK(
          half_rotary_dim % vec_size == 0,
          "half rotary dimension must be divisible by selected vec_size for neox style, got half_rotary_dim=",
          half_rotary_dim,
          ", vec_size=",
          vec_size);
    }

    if (half_rotary_dim >= vec_size) {
      const int64_t half_rotary_vecs = half_rotary_dim / vec_size;
      const int64_t lane_xor = half_rotary_vecs % NUM_REDUCE_STAGES;
      TORCH_CHECK(
          lane_xor == 0 || ((lane_xor & (lane_xor - 1)) == 0 && lane_xor < NUM_REDUCE_STAGES),
          "neox style requires a power-of-two subgroup lane xor, got ",
          lane_xor);
    }
  }
}

template <bool interleave, typename scalar_t, typename weight_t>
void launch_fused_qk_norm_rope_core(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    double base,
    torch::Tensor& position_ids,
    double factor,
    double low,
    double high,
    double attention_factor,
    int64_t rotary_dim,
    const char* op_name) {
  const auto layout = qknorm::make_packed_qkv_layout_from_ptr<scalar_t, weight_t>(
      static_cast<scalar_t*>(qkv.data_ptr()),
      static_cast<const weight_t*>(q_weight.data_ptr()),
      static_cast<const weight_t*>(k_weight.data_ptr()),
      qkv.size(0),
      num_heads_q,
      num_heads_k,
      num_heads_v,
      head_dim);

  const int vec_size =
      qknorm::vec_size<scalar_t, weight_t>(layout, qknorm::max_vec_size<scalar_t>(), NUM_REDUCE_STAGES);
  const bool is_warp = (head_dim == 64 || head_dim == 128 || head_dim == 256);
  if (is_warp) {
    validate_rotary_for_vec_size(head_dim, rotary_dim, !interleave, vec_size);
  } else {
    TORCH_CHECK(
        rotary_dim > 0 && rotary_dim <= head_dim, op_name, ": rotary_dim must be in (0, head_dim], got ", rotary_dim);
    TORCH_CHECK(rotary_dim % 2 == 0, op_name, ": rotary_dim must be even for RoPE, got ", rotary_dim);
  }

  const QKNormRopePostOp<interleave> post_op{
      position_ids.data_ptr<int>(),
      static_cast<float>(std::log2(base)),
      static_cast<float>(factor),
      static_cast<float>(low),
      static_cast<float>(high),
      static_cast<float>(attention_factor),
      static_cast<int>(rotary_dim)};

  qknorm::dispatch_head_dim<scalar_t, weight_t>(
      layout, static_cast<qknorm::acc_type_t<scalar_t>>(eps), post_op, op_name, true);
}

template <bool interleave, typename scalar_t>
void dispatch_fused_qk_norm_rope_weight_type(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    double base,
    torch::Tensor& position_ids,
    double factor,
    double low,
    double high,
    double attention_factor,
    int64_t rotary_dim,
    const char* op_name) {
#define QKNORM_ROPE_WEIGHT_CASE(enum_type, cpp_type)                \
  case enum_type: {                                                 \
    using weight_t = cpp_type;                                      \
    launch_fused_qk_norm_rope_core<interleave, scalar_t, weight_t>( \
        qkv,                                                        \
        num_heads_q,                                                \
        num_heads_k,                                                \
        num_heads_v,                                                \
        head_dim,                                                   \
        eps,                                                        \
        q_weight,                                                   \
        k_weight,                                                   \
        base,                                                       \
        position_ids,                                               \
        factor,                                                     \
        low,                                                        \
        high,                                                       \
        attention_factor,                                           \
        rotary_dim,                                                 \
        op_name);                                                   \
    break;                                                          \
  }

  switch (q_weight.scalar_type()) {
    QKNORM_ROPE_WEIGHT_CASE(at::ScalarType::Double, double)
    QKNORM_ROPE_WEIGHT_CASE(at::ScalarType::Float, float)
    QKNORM_ROPE_WEIGHT_CASE(at::ScalarType::Half, decltype(c10::impl::ScalarTypeToCPPType<at::ScalarType::Half>::t))
    QKNORM_ROPE_WEIGHT_CASE(
        at::ScalarType::BFloat16, decltype(c10::impl::ScalarTypeToCPPType<at::ScalarType::BFloat16>::t))
    QKNORM_ROPE_WEIGHT_CASE(at::ScalarType::Float8_e4m3fn, float_e4m3_t)
    default:
      TORCH_CHECK(false, op_name, " not implemented for weight type '", toString(q_weight.scalar_type()), "'");
  }

#undef QKNORM_ROPE_WEIGHT_CASE
}

template <bool interleave>
void dispatch_fused_qk_norm_rope_qkv_type(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    double base,
    torch::Tensor& position_ids,
    double factor,
    double low,
    double high,
    double attention_factor,
    int64_t rotary_dim,
    const char* op_name) {
#define QKNORM_ROPE_QKV_CASE(enum_type, cpp_type)                  \
  case enum_type: {                                                \
    using scalar_t = cpp_type;                                     \
    dispatch_fused_qk_norm_rope_weight_type<interleave, scalar_t>( \
        qkv,                                                       \
        num_heads_q,                                               \
        num_heads_k,                                               \
        num_heads_v,                                               \
        head_dim,                                                  \
        eps,                                                       \
        q_weight,                                                  \
        k_weight,                                                  \
        base,                                                      \
        position_ids,                                              \
        factor,                                                    \
        low,                                                       \
        high,                                                      \
        attention_factor,                                          \
        rotary_dim,                                                \
        op_name);                                                  \
    break;                                                         \
  }

  switch (qkv.scalar_type()) {
    QKNORM_ROPE_QKV_CASE(at::ScalarType::Half, decltype(c10::impl::ScalarTypeToCPPType<at::ScalarType::Half>::t))
    QKNORM_ROPE_QKV_CASE(
        at::ScalarType::BFloat16, decltype(c10::impl::ScalarTypeToCPPType<at::ScalarType::BFloat16>::t))
    QKNORM_ROPE_QKV_CASE(at::ScalarType::Float8_e4m3fn, float_e4m3_t)
    default:
      TORCH_CHECK(false, op_name, " not implemented for qkv type '", toString(qkv.scalar_type()), "'");
  }

#undef QKNORM_ROPE_QKV_CASE
}

void fused_qk_norm_rope(
    torch::Tensor& qkv,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t num_heads_v,
    int64_t head_dim,
    double eps,
    torch::Tensor& q_weight,
    torch::Tensor& k_weight,
    double base,
    bool is_neox,
    torch::Tensor& position_ids,
    double factor,
    double low,
    double high,
    double attention_factor,
    int64_t rotary_dim) {
  constexpr const char* op_name = "fused_qk_norm_rope";

  TORCH_CHECK(qkv.dim() == 2, op_name, ": qkv must be 2D [num_tokens, total_qkv_heads * head_dim]");
  TORCH_CHECK(position_ids.dim() == 1, op_name, ": position_ids must be 1D [num_tokens]");
  TORCH_CHECK(q_weight.dim() == 1, op_name, ": q_weight must be 1D [head_dim]");
  TORCH_CHECK(k_weight.dim() == 1, op_name, ": k_weight must be 1D [head_dim]");
  TORCH_CHECK(q_weight.size(0) == head_dim, op_name, ": q_weight size must match head_dim");
  TORCH_CHECK(k_weight.size(0) == head_dim, op_name, ": k_weight size must match head_dim");
  TORCH_CHECK(q_weight.scalar_type() == k_weight.scalar_type(), op_name, ": q_weight and k_weight dtype must match");
  TORCH_CHECK(head_dim > 0, op_name, ": head_dim must be positive, got ", head_dim);

  CHECK_DEVICE(qkv);
  CHECK_CONTIGUOUS(qkv);
  CHECK_DEVICE(position_ids);
  CHECK_CONTIGUOUS(position_ids);
  CHECK_DEVICE(q_weight);
  CHECK_CONTIGUOUS(q_weight);
  CHECK_DEVICE(k_weight);
  CHECK_CONTIGUOUS(k_weight);

  TORCH_CHECK(
      position_ids.scalar_type() == at::ScalarType::Int,
      op_name,
      ": position_ids must have dtype int32 (at::kInt); got ",
      position_ids.scalar_type());

  const int64_t num_tokens = qkv.size(0);
  TORCH_CHECK(position_ids.size(0) == num_tokens, op_name, ": position_ids length must match qkv num_tokens");

  const int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
  TORCH_CHECK(qkv.size(1) == total_heads * head_dim, op_name, ": qkv size must match total heads and head_dim");

  if (is_neox) {
    dispatch_fused_qk_norm_rope_qkv_type<false>(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
        op_name);
  } else {
    dispatch_fused_qk_norm_rope_qkv_type<true>(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        base,
        position_ids,
        factor,
        low,
        high,
        attention_factor,
        rotary_dim,
        op_name);
  }
}

void fused_inplace_qknorm(
    torch::Tensor& q, torch::Tensor& k, torch::Tensor& q_weight, torch::Tensor& k_weight, double eps) {
  TORCH_CHECK(q.dim() == 3, "fused_inplace_qknorm: q must be 3D [num_tokens, num_q_heads, head_dim]");
  TORCH_CHECK(k.dim() == 3, "fused_inplace_qknorm: k must be 3D [num_tokens, num_k_heads, head_dim]");
  TORCH_CHECK(q.size(0) == k.size(0), "fused_inplace_qknorm: q and k must have same num_tokens");
  TORCH_CHECK(q.size(2) == k.size(2), "fused_inplace_qknorm: q and k must have same head_dim");
  TORCH_CHECK(q_weight.dim() == 1 && k_weight.dim() == 1, "fused_inplace_qknorm: weights must be 1D");
  TORCH_CHECK(q_weight.size(0) == q.size(2), "fused_inplace_qknorm: q_weight size must match head_dim");
  TORCH_CHECK(k_weight.size(0) == k.size(2), "fused_inplace_qknorm: k_weight size must match head_dim");
  TORCH_CHECK(q.stride(2) == 1 && k.stride(2) == 1, "fused_inplace_qknorm: q/k last dimension must be contiguous");
  CHECK_DEVICE(q);
  CHECK_DEVICE(k);
  CHECK_DEVICE(q_weight);
  CHECK_DEVICE(k_weight);
  TORCH_CHECK(q.scalar_type() == k.scalar_type(), "fused_inplace_qknorm: q and k dtype must match");
  TORCH_CHECK(
      q_weight.scalar_type() == k_weight.scalar_type(), "fused_inplace_qknorm: q_weight and k_weight dtype must match");

  SYCL_DISPATCH_FLOATING_TYPES(
      at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "fused_inplace_qknorm", [&]() {
        SYCL_DISPATCH_WEIGHT_TYPES(
            at::ScalarType::Half, at::ScalarType::BFloat16, q_weight.scalar_type(), "fused_inplace_qknorm", [&]() {
              const auto layout = qknorm::make_separated_layout<scalar_t, weight_t>(q, k, q_weight, k_weight);
              qknorm::dispatch_head_dim<scalar_t, weight_t>(
                  layout,
                  static_cast<qknorm::acc_type_t<scalar_t>>(eps),
                  qknorm::NoPostOp{},
                  "fused_inplace_qknorm",
                  true);
            });
      });
}

}  // namespace at::native::xpu
